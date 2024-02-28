import copy
import torch
import torch.nn as nn
import numpy as np
from torch.nn.init import normal_
import torch.nn.functional as F
from mmdet.models.utils.builder import TRANSFORMER
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmcv.runner.base_module import BaseModule, ModuleList, Sequential
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from torchvision.transforms.functional import rotate
from mmdet3d.bevformer.modules.temporal_self_attention import TemporalSelfAttention
from mmdet3d.bevformer.modules.spatial_cross_attention import MSDeformableAttention3D
from mmdet3d.bevformer.modules.decoder import CustomMSDeformableAttention
from mmdet3d.models.builder import build_fuser, FUSERS
from mmcv.runner import force_fp32, auto_fp16
from typing import List

@FUSERS.register_module()
class ConvFuser(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__(
            nn.Conv2d(sum(in_channels), out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return super().forward(torch.cat(inputs, dim=1))

@TRANSFORMER.register_module()
class MapTRPerceptionTransformer(BaseModule):
    """Implements the Detr3D transformer.
    Args:
        as_two_stage (bool): Generate query from encoder features.
            Default: False.
        num_feature_levels (int): Number of feature maps from FPN:
            Default: 4.
        two_stage_num_proposals (int): Number of proposals when set
            `as_two_stage` as True. Default: 300.
    """

    def __init__(self,
                 num_feature_levels=4,
                 num_cams=6,
                 z_cfg=dict(
                    pred_z_flag=False,
                    gt_z_flag=False,
                 ),
                 two_stage_num_proposals=300,
                 fuser=None,
                 encoder=None,
                 decoder=None,
                 embed_dims=256,
                 rotate_prev_bev=True,
                 use_fast_perception_neck=True,
                 num_decoders=3,
                 use_shift=True,
                 use_can_bus=True,
                 can_bus_norm=True,
                 use_cams_embeds=True,
                 rotate_center=[100, 100],
                 modality='vision',
                 feat_down_sample_indice=-1,
                 **kwargs):
        super(MapTRPerceptionTransformer, self).__init__(**kwargs)
        if modality == 'fusion':
            self.fuser = build_fuser(fuser) #TODO
        # self.use_attn_bev = encoder['type'] == 'BEVFormerEncoder'
        self.use_attn_bev = 'BEVFormerEncoder' in encoder['type']
        self.encoder = build_transformer_layer_sequence(encoder)
        if use_fast_perception_neck:
            self.decoder = nn.ModuleList()
            for i in range(num_decoders):
                self.decoder.append(build_transformer_layer_sequence(decoder))
        else:
            self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = embed_dims
        self.num_feature_levels = num_feature_levels
        self.num_cams = num_cams
        self.use_fast_perception_neck = use_fast_perception_neck
        self.fp16_enabled = False

        self.rotate_prev_bev = rotate_prev_bev
        self.use_shift = use_shift
        self.use_can_bus = use_can_bus
        self.can_bus_norm = can_bus_norm
        self.use_cams_embeds = use_cams_embeds

        self.two_stage_num_proposals = two_stage_num_proposals
        self.z_cfg=z_cfg
        self.init_layers()
        self.rotate_center = rotate_center
        self.feat_down_sample_indice = feat_down_sample_indice
        self.compress_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

    def init_layers(self):
        """Initialize layers of the Detr3DTransformer."""
        self.level_embeds = nn.Parameter(torch.Tensor(
            self.num_feature_levels, self.embed_dims))
        self.cams_embeds = nn.Parameter(
            torch.Tensor(self.num_cams, self.embed_dims))
        self.reference_points = nn.Linear(self.embed_dims, 2) if not self.z_cfg['gt_z_flag'] \
                            else nn.Linear(self.embed_dims, 3)
        self.can_bus_mlp = nn.Sequential(
            nn.Linear(18, self.embed_dims // 2),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims // 2, self.embed_dims),
            nn.ReLU(inplace=True),
        )
        if self.can_bus_norm:
            self.can_bus_mlp.add_module('norm', nn.LayerNorm(self.embed_dims))

    def init_weights(self):
        """Initialize the transformer weights."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformableAttention3D) or isinstance(m, TemporalSelfAttention) \
                    or isinstance(m, CustomMSDeformableAttention):
                try:
                    m.init_weight()
                except AttributeError:
                    m.init_weights()
        normal_(self.level_embeds)
        normal_(self.cams_embeds)
        xavier_init(self.reference_points, distribution='uniform', bias=0.)
        xavier_init(self.can_bus_mlp, distribution='uniform', bias=0.)

    def lss_bev_encode(
            self,
            mlvl_feats,
            prev_bev=None,
            **kwargs):
        # import ipdb;ipdb.set_trace()
        # assert len(mlvl_feats) == 1, 'Currently we only use last single level feat in LSS'
        # import ipdb;ipdb.set_trace()
        images = mlvl_feats[self.feat_down_sample_indice]
        img_metas = kwargs['img_metas']
        encoder_outputdict = self.encoder(images,img_metas)
        bev_embed = encoder_outputdict['bev']
        depth = encoder_outputdict['depth']
        bs, c, _,_ = bev_embed.shape
        bev_embed = bev_embed.view(bs,c,-1).permute(0,2,1).contiguous()
        ret_dict = dict(
            bev=bev_embed,
            depth=depth
        )
        return ret_dict

    def get_bev_features(
            self,
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=[0.512, 0.512],
            bev_pos=None,
            prev_bev=None,
            **kwargs):
        """
        obtain bev features.
        """
        ret_dict = self.lss_bev_encode(
            mlvl_feats,
            prev_bev=prev_bev,
            **kwargs)
        bev_embed = ret_dict['bev']
        depth = ret_dict['depth']
        if lidar_feat is not None:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
            lidar_feat = lidar_feat.permute(0,1,3,2).contiguous() # B C H W
            lidar_feat = nn.functional.interpolate(lidar_feat, size=(bev_h, bev_w), mode='bicubic', align_corners=False)
            bev_embed = self.fuser([bev_embed, lidar_feat])
        else:
            bs = mlvl_feats[0].size(0)
            bev_embed = bev_embed.view(bs, bev_h, bev_w, -1).permute(0,3,1,2).contiguous()
        ret_dict = dict(
            bev=bev_embed,
            depth=depth
        )
        return ret_dict

    def format_feats(self, mlvl_feats):
        bs = mlvl_feats[0].size(0)

        feat_flatten = []
        spatial_shapes = []
        for lvl, feat in enumerate(mlvl_feats):
            bs, num_cam, c, h, w = feat.shape
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)

            if self.use_cams_embeds:
                feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None,
                                            None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)

        feat_flatten = torch.cat(feat_flatten, 2)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=feat.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros(
            (1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))

        feat_flatten = feat_flatten.permute(
            0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims)
        return feat_flatten, spatial_shapes, level_start_index

    # TODO apply fp16 to this module cause grad_norm NAN
    # @auto_fp16(apply_to=('mlvl_feats', 'bev_queries', 'object_query_embed', 'prev_bev', 'bev_pos'))
    @force_fp32(apply_to=('mlvl_feats', 'lidar_feat'))
    def forward(self,
                mlvl_feats,
                lidar_feat,
                bev_queries,
                object_query_embed,
                bev_h,
                bev_w,
                grid_length=[0.512, 0.512],
                bev_pos=None,
                reg_branches=None,
                cls_branches=None,
                backbone=None,
                neck=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """

        ouput_dic = self.get_bev_features(
            mlvl_feats,
            lidar_feat,
            bev_queries,
            bev_h,
            bev_w,
            grid_length=grid_length,
            bev_pos=bev_pos,
            prev_bev=prev_bev,
            **kwargs)  # bev_embed shape: bs, bev_h*bev_w, embed_dims

        bev_embed = ouput_dic['bev']
        depth = ouput_dic['depth']
        if self.use_fast_perception_neck:
            depth_list = []
            inter_states_list = []
            init_reference_out_list = []
            inter_references_out_list = []
            bev_embeds = neck(bev_embed) # list 上采样分支数量

            for index, bev_embed in enumerate(bev_embeds):
                bev_embed = bev_embed.flatten(2).permute(0, 2, 1).contiguous().float()
                bs = mlvl_feats[0].size(0)
                query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
                query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
                query = query.unsqueeze(0).expand(bs, -1, -1)
                reference_points = self.reference_points(query_pos)
                reference_points = reference_points.sigmoid()
                init_reference_out = reference_points

                query = query.permute(1, 0, 2)
                query_pos = query_pos.permute(1, 0, 2)
                bev_embed = bev_embed.permute(1, 0, 2)

                feat_flatten, feat_spatial_shapes, feat_level_start_index = self.format_feats(mlvl_feats)

                inter_states, inter_references = self.decoder[index](
                    query=query,
                    key=None,
                    value=bev_embed,
                    query_pos=query_pos,
                    reference_points=reference_points,
                    reg_branches=reg_branches,
                    cls_branches=cls_branches,
                    spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                    level_start_index=torch.tensor([0], device=query.device),
                    mlvl_feats=mlvl_feats,
                    feat_flatten=feat_flatten,
                    feat_spatial_shapes=feat_spatial_shapes,
                    feat_level_start_index=feat_level_start_index,
                    **kwargs)

                inter_references_out = inter_references
                inter_states_list.append(inter_states)
                init_reference_out_list.append(init_reference_out)
                inter_references_out_list.append(inter_references_out)
            return bev_embeds, depth, inter_states_list, init_reference_out_list, inter_references_out_list  # depth不需要list
        else:
            #bev_embed = neck(backbone(bev_embed))[0].float()
            #bev_embed = self.compress_layer(bev_embed)
            bev_embed_ = bev_embed.flatten(2).permute(0, 2, 1).contiguous().float()
            depth = ouput_dic['depth']
            bs = mlvl_feats[0].size(0)
            query_pos, query = torch.split(object_query_embed, self.embed_dims, dim=1)
            query_pos = query_pos.unsqueeze(0).expand(bs, -1, -1)
            query = query.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_pos)
            reference_points = reference_points.sigmoid()
            init_reference_out = reference_points

            query = query.permute(1, 0, 2)
            query_pos = query_pos.permute(1, 0, 2)
            bev_embed_ = bev_embed_.permute(1, 0, 2)

            feat_flatten, feat_spatial_shapes, feat_level_start_index = self.format_feats(mlvl_feats)

            inter_states, inter_references = self.decoder(
                query=query,
                key=None,
                value=bev_embed_,
                query_pos=query_pos,
                reference_points=reference_points,
                reg_branches=reg_branches,
                cls_branches=cls_branches,
                spatial_shapes=torch.tensor([[bev_h, bev_w]], device=query.device),
                level_start_index=torch.tensor([0], device=query.device),
                mlvl_feats=mlvl_feats,
                feat_flatten=feat_flatten,
                feat_spatial_shapes=feat_spatial_shapes,
                feat_level_start_index=feat_level_start_index,
                **kwargs)
            inter_references_out = inter_references

            return bev_embed, depth, inter_states, init_reference_out, inter_references_out
