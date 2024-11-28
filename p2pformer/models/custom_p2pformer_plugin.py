import torch
import torch.nn as nn
import torch.nn.functional as F

from mmcv.runner import BaseModule
from mmcv.cnn import xavier_init, PLUGIN_LAYERS, build_norm_layer
from mmcv.cnn.bricks.registry import TRANSFORMER_LAYER_SEQUENCE
from mmcv.cnn.bricks.transformer import (TransformerLayerSequence,
                                         build_transformer_layer_sequence)

@TRANSFORMER_LAYER_SEQUENCE.register_module(force=True)
class LineDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 hidden_dim=256,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=True,
                 update_pos_embed=True,
                 **kwargs):

        super(LineDecoder, self).__init__(*args, **kwargs)
        self.hidden_dim = hidden_dim
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None
        self.update_pos_embed = update_pos_embed
        if self.update_pos_embed:
            self.pos_off_predictor = MLP(self.hidden_dim, self.hidden_dim * 2,
                                         self.hidden_dim, 3)

    def forward(self,
                query,
                *args,
                key,
                value,
                key_pos,
                query_pos=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        num_level = len(value)
        per_layer_per_level = len(self.layers) // num_level
        intermediate = []
        intermediate_query_pos = []
        for lid, layer in enumerate(self.layers):
            query = layer(query,
                          *args,
                          query_pos=query_pos,
                          key=value[lid // per_layer_per_level],
                          value=value[lid // per_layer_per_level],
                          key_pos=key_pos[lid // per_layer_per_level],
                          **kwargs)
            if self.update_pos_embed:
                pos_off = self.pos_off_predictor(query)
                query_pos = query_pos + pos_off
            intermediate_query_pos.append(query_pos)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        return torch.stack(intermediate), torch.stack(intermediate_query_pos)

@TRANSFORMER_LAYER_SEQUENCE.register_module(force=True)
class OrderDecoder(TransformerLayerSequence):
    """Implements the decoder in DETR transformer.

    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self,
                 *args,
                 post_norm_cfg=dict(type='LN'),
                 return_intermediate=False,
                 **kwargs):
        super(OrderDecoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate
        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(post_norm_cfg,
                                              self.embed_dims)[1]
        else:
            self.post_norm = None

    def forward(self,
                query,
                *args,
                key,
                value,
                key_pos,
                query_pos=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_query, bs, embed_dims)`.

        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        intermediate = []
        for lid, layer in enumerate(self.layers):
            query = layer(query,
                          *args,
                          query_pos=query_pos,
                          key=value,
                          value=value,
                          key_pos=key_pos,
                          **kwargs)
            if self.return_intermediate:
                if self.post_norm is not None:
                    intermediate.append(self.post_norm(query))
                else:
                    intermediate.append(query)
        if self.return_intermediate:
            return torch.stack(intermediate)
        else:
            return query


@PLUGIN_LAYERS.register_module(force=True)
class LinePredictor(BaseModule):
    def __init__(self,
                 num_query=30,
                 out_channel=4,
                 roi_wh=(40, 40),
                 refer_points_nums=36,
                 pred_line_with_pos=True,
                 update_pos_embed=True,
                 mode='line',
                 is_detach_order_grad=True,
                 line_decoder=None,
                 order_decoder=None,
                 init_cfg=None,
                 **kwargs):
        super(LinePredictor, self).__init__(init_cfg=init_cfg)
        self.is_detach_order_grad = is_detach_order_grad
        self.num_query = num_query
        self.out_channel = out_channel
        self.roi_w = roi_wh[0]
        self.roi_h = roi_wh[1]
        self.refer_points_nums = refer_points_nums
        line_decoder.update({'update_pos_embed': update_pos_embed})
        self.line_decoder = build_transformer_layer_sequence(line_decoder)
        self.order_decoder = build_transformer_layer_sequence(order_decoder)
        self.num_layers = self.line_decoder.num_layers
        self.hidden_dim = self.line_decoder.hidden_dim
        self.pred_line_with_pos = pred_line_with_pos
        self.update_pos_embed = update_pos_embed
        self.mode = mode
        if self.mode == 'point':
            self.out_channel = self.out_channel // 3

        self._init_layers()

    def _init_layers(self):
        if self.pred_line_with_pos:
            self.predictor_line = MLP(self.hidden_dim * 2, self.hidden_dim * 2,
                                      self.out_channel, 3)
        else:
            self.predictor_line = MLP(self.hidden_dim, self.hidden_dim * 2,
                                      self.out_channel, 3)
        self.predictor_score = nn.Linear(self.hidden_dim, 2)
        if self.mode == 'point':
            self.converger_query = nn.Linear(self.hidden_dim * 3, self.hidden_dim)

        self.predictor_order = nn.Linear(self.hidden_dim, self.refer_points_nums)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self,
                query,
                query_pos,
                mlvl_feats,
                mlvl_pos_embeds,
                shuffle_idxs=None,
                **kwargs):
        """Forward function for 'LinePredictor'.

        Args:
            query (Tensor): Initialized queries. Shape [nq, ngt, c].
            query_pos (Tensor): Initialized queries' pos_embeds.
                Shape [nq, ngt, c].
            mlvl_feats (list[Tensor]): Input multi-level roi features.
                Each element has shape [h*w, ngt, c].
            mlvl_pos_embeds (list[Tensor]): Corresponding pos_embeds of
                mlvl_feats. Each element has shape [h*w, ngt, c].
        """
        bs, nq = mlvl_feats[0].size(1), query.size(0)
        # hs (num_layers, nq, ngt, c)
        hs, pos_embeds = self.line_decoder(
            query=query,
            key=mlvl_feats,
            value=mlvl_feats,
            key_pos=mlvl_pos_embeds,
            query_pos=query_pos,
            key_padding_mask=None)

        if self.pred_line_with_pos:
            queries_with_pos = torch.cat([hs, pos_embeds], dim=-1) # (nl, nq, ngt, 2c)
        else:
            queries_with_pos = hs
        if self.mode == 'point':
            nl, nq, ngt, _ = hs.size()
            hs = hs.reshape(nl, nq // 3, 3, ngt, _)
            hs = hs.permute(0, 1, 3, 2, 4).flatten(3)
            hs = self.converger_query(hs)
        line_preds = self.predictor_line(queries_with_pos).transpose(1, 2) # (nl, ngt, nq, 4)
        line_preds = line_preds.sigmoid()
        line_scores = self.predictor_score(hs).transpose(1, 2) # (nl, ngt, nq, 2)
        # pos_off = self.pos_off_predictor(hs) # (nl, nq, ngt, c)

        if self.mode == 'point':
            nl, ngt, nq, _ = line_preds.size()
            line_preds = line_preds.reshape(nl, ngt, nq // 3, 3, _)
            line_preds = line_preds.flatten(3)

        if self.is_detach_order_grad:
            query_order = hs[-1].detach()
        else:
            query_order = hs[-1]
        query_order = self.order_decoder(
            query=query_order,
            key=query_order,
            value=query_order,
            key_pos=None,
            query_pos=None,
            key_padding_mask=None,
        )
        order_preds = self.predictor_order(query_order).transpose(0, 1)
        return line_preds, line_scores, order_preds

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

