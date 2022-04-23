import argparse
from abc import ABC
from collections import defaultdict

import torch
from torch import nn
from .backbone.transformer import Net, MultiheadAttn, PositionalEncoding, single_object_encoder, MLP
from .utils import get_siamese_features, build_module


class _AbstractCoAttentionModule(nn.Module):
    side = None

    def forward(self, lang, obj, lang_mask, obj_mask, co_mask):
        raise NotImplementedError


class _AbstractSAModule(_AbstractCoAttentionModule, ABC):
    def __init__(self, dim, dim_ff, n_head, msa_dropout, ffn_dropout):
        super().__init__()
        self.dim = dim
        self.msa = MultiheadAttn(dim, n_head, dropout=msa_dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim_ff, dim)
        )

    def _forward(self, q, k, v, mask, add_q=True):
        msa = self.msa(q, k, v, mask)
        if add_q:
            x = self.norm1(q + msa)
        else:
            x = self.norm1(v + msa)
        x = x + self.ffn(x)
        x = self.norm2(x)
        return x


class LangSaModule(_AbstractSAModule):
    side = 'lang'

    def forward(self, lang, obj, lang_mask, obj_mask, co_mask):
        x = lang
        mask = lang_mask
        return self._forward(x, x, x, mask)


class ObjSaModule(_AbstractSAModule):
    side = 'obj'

    def forward(self, lang, obj, lang_mask, obj_mask, co_mask):
        x = obj
        mask = obj_mask
        return self._forward(x, x, x, mask)


class LangGaModule(_AbstractSAModule):
    side = 'lang'

    def forward(self, lang, obj, lang_mask, obj_mask, co_mask):
        x_value = lang
        x_guide = obj
        return self._forward(x_value, x_guide, x_guide, co_mask)


class ObjGaModule(_AbstractSAModule):
    side = 'obj'

    def forward(self, lang, obj, lang_mask, obj_mask, co_mask):
        x_value = obj
        x_guide = lang
        return self._forward(x_value, x_guide, x_guide, co_mask.transpose(-1, -2))


class ObjEdgeGaModule(_AbstractCoAttentionModule):
    side = 'obj'

    def __init__(self, dim, dim_ff, n_head, msa_dropout, ffn_dropout):
        super().__init__()
        self.dim = dim
        self.msa_edge = MultiheadAttn(dim, n_head, dropout=msa_dropout)
        self.msa_node = MultiheadAttn(dim, n_head, dropout=msa_dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.norme = nn.LayerNorm(dim)
        self.normq = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim * 2, dim_ff),
            nn.ReLU(),
            nn.Dropout(ffn_dropout),
            nn.Linear(dim_ff, dim)
        )

    def _forward(self, q, node, edge):
        xe = self.norme(q + edge)
        xn = self.normq(q + node)
        x = torch.cat((xn, xe), -1)
        x = xn + self.ffn(x)
        x = self.norm2(x)
        return x

    def _forward_edge(self, lang, obj, lang_mask, obj_mask, co_mask):
        x_value = obj  # N nobj ndim
        x_value = x_value.unsqueeze(2)
        x_expand_t = x_value.transpose(1, 2)  # N 1 nobj ndim
        x_value = x_value - x_expand_t  # N nobj nobj ndim

        N, nobj, _, ndim = x_value.shape
        x_value = x_value.transpose(0, 1).reshape(-1, nobj, ndim)

        x_guide = lang.repeat(nobj, 1, 1)

        mask = co_mask.transpose(-1, -2).repeat(nobj, 1, 1)
        edge = self.msa_edge(x_value, x_guide, x_guide, mask)
        edge = edge.max(1)[0]
        edge = edge.reshape(nobj, N, ndim).transpose(0, 1)
        return edge

    def _forward_node(self, lang, obj, lang_mask, obj_mask, co_mask):
        x_value = obj
        x_guide = lang
        return self.msa_node(x_value, x_guide, x_guide, co_mask.transpose(-1, -2))

    def forward(self, lang, obj, *args):
        node = self._forward_node(lang, obj, *args)
        edge = self._forward_edge(lang, obj, *args)
        return self._forward(obj, node, edge)


class CoAttentionLayer(_AbstractCoAttentionModule):
    def __init__(self, lang_layer, obj_layer):
        super().__init__()
        self.obj_layer = obj_layer
        self.lang_layer = lang_layer
        assert self.obj_layer is None or self.obj_layer.side == 'obj'
        assert self.lang_layer is None or self.lang_layer.side == 'lang'

    def forward(self, *args):
        lang, obj = args[:2]
        if self.obj_layer:
            obj = self.obj_layer(*args)
        if self.lang_layer:
            lang = self.lang_layer(*args)
        return lang, obj


def instantiate_stacking(args: argparse.Namespace, vocab, n_obj_classes: int, model_type) -> nn.Module:
    dim = args.D
    obj_enc = single_object_encoder(dim)
    word_enc = nn.Sequential(
        nn.Embedding(len(vocab), dim, padding_idx=vocab.pad),
        nn.Dropout(args.dropout),
        nn.Linear(dim, dim),
        PositionalEncoding(dim)
    )

    depth = args.depth
    if model_type == 'transrefer3d':
        layers = sum([[
            (LangSaModule(dim, dim, 4, 0.05, args.dropout), ObjSaModule(dim, dim, 4, 0.05, args.dropout)),
            (LangGaModule(dim, dim, 4, 0.05, args.dropout), ObjEdgeGaModule(dim, dim, 4, 0.05, args.dropout)),
        ] for _ in range(depth)], [])
    else:
        raise ValueError(f'unsupported model {model_type}')
    layers = [CoAttentionLayer(*models) for models in layers]
    obj_clf = MLP(dim, [128, 256, n_obj_classes], dropout_rate=0.15)
    lang_clf = MLP(dim, [128, 256, n_obj_classes], dropout_rate=0.2)
    object_language_clf = MLP(dim * 2, out_channels=[128, 64, 1], dropout_rate=0.05)
    return Net(args, obj_enc, word_enc, layers, obj_clf, lang_clf, object_language_clf, dropout=args.dropout)
