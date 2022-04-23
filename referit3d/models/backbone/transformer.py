import math

import torch
import torch.nn.functional as F
import argparse
from torch import nn
from collections import defaultdict

from ..default_blocks import *
from ..utils import get_siamese_features
from ...in_out.vocabulary import Vocabulary
from .mlp import MLP

try:
    from . import PointNetPP
except ImportError:
    PointNetPP = None


class MultiheadAttn(nn.Module):
    def __init__(self, dim, nhead, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.nhead = nhead
        self.head_dim = int(dim // nhead)
        assert self.nhead * self.head_dim == self.dim
        self.linears = nn.ModuleList([nn.Linear(dim, dim) for _ in range(4)])
        if dropout == 0:
            self.dropout = None
        else:
            self.dropout = nn.Dropout(dropout)

    def attention(self, queries, keys, values, mask=None, dropout=None):
        """
            queries: B x H x S x headdim
            keys: B x H x L x headdim
            values: B x H x L x headdim
            mask: B x 1 x S x L
        """
        headdim = queries.size(-1)
        scores = queries @ keys.transpose(-1, -2) / math.sqrt(headdim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        scores = torch.softmax(scores, dim=-1)
        if dropout is not None:
            scores = dropout(scores)
        return scores @ values

    def forward(self, query, key, value, mask=None, sum_seq=False):
        """
            query: B x S x D
            key: B x L x D
            value: B x L x D
            mask: B x S x L
        """
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)
        queries, keys, values = [
            layer(x).view(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)
            for layer, x in zip(self.linears[:3], (query, key, value))
        ]
        result = self.attention(queries, keys, values, mask, self.dropout)
        if sum_seq:
            result = result.sum(2, keepdim=True)
        result = result.transpose(1, 2).contiguous().view(batch_size, -1, self.dim)
        return self.linears[-1](result)


class CatFuse(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.linear = nn.Linear(dim * 2, dim)

    def forward(self, feat_1, feat_2):
        return self.linear(torch.cat((feat_1, feat_2), dim=-1))


class CoAttentionNet(nn.Module):
    def __init__(self, dim, dim_ff, lang_head, coattn_head, gnn, dropout=0.1):
        super().__init__()
        self.dim = dim
        self.lang_head = lang_head
        self.coattn_head = coattn_head
        self.dropout = dropout

        # language side
        self.self_attn = MultiheadAttn(dim, lang_head, 0)
        self.lang_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])
        self.lang_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.rel_attn = MultiheadAttn(dim, 1, 0)
        self.ent_attn = MultiheadAttn(dim, 1, 0)
        self.lang_ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim)
        )

        # object side
        self.gnn = gnn
        self.obj_dropouts = nn.ModuleList([nn.Dropout(dropout) for _ in range(3)])
        self.obj_norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(3)])
        self.obj_ff = nn.Sequential(
            nn.Linear(dim, dim_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, dim)
        )

        # co-attention side
        self.attn_lang = MultiheadAttn(dim, coattn_head, dropout)
        self.attn_obj = MultiheadAttn(dim, coattn_head, dropout)
        self.lang_fusion = CatFuse(dim)
        self.obj_fusion = CatFuse(dim)

    def forward(self, lang_feat, obj_feat, lang_mask, obj_mask, co_mask):
        """
            lang_feat: B x nword x D
            obj_feat: B x nobj x D
            lang_mask: B x nword x nword
            obj_mask: B x nobj x nobj
            co_mask: B x nword x nobj
        """
        # encode
        lang_feat_self_attn = self.self_attn(lang_feat, lang_feat, lang_feat, mask=lang_mask)
        lang_feat_rel = self.rel_attn(lang_feat, lang_feat, lang_feat, mask=lang_mask).max(dim=1)[0]
        lang_feat_ent = self.ent_attn(lang_feat, lang_feat, lang_feat, mask=lang_mask).max(dim=1)[0]
        lang_feat = lang_feat + self.lang_dropouts[0](lang_feat_self_attn)
        lang_feat = self.lang_norms[0](lang_feat)

        obj_feat_gnn = self.gnn(obj_feat, lang_feat_ent, lang_feat_rel)
        obj_feat = obj_feat + self.obj_dropouts[0](obj_feat_gnn)
        obj_feat = self.obj_norms[0](obj_feat)

        # coattention
        attn_lang_feat = self.attn_lang(obj_feat, lang_feat, lang_feat, mask=co_mask.transpose(-1, -2))
        attn_obj_feat = self.attn_obj(lang_feat, obj_feat, obj_feat, mask=co_mask)

        lang_feat = self.lang_fusion(lang_feat, self.lang_dropouts[1](attn_obj_feat))
        lang_feat = self.lang_norms[1](lang_feat)
        obj_feat = self.obj_fusion(obj_feat, self.obj_dropouts[1](attn_lang_feat))
        obj_feat = self.obj_norms[1](obj_feat)

        # feed forward
        lang_feat_ff = self.lang_ff(lang_feat)
        lang_feat = lang_feat + self.lang_dropouts[2](lang_feat_ff)
        lang_feat = self.lang_norms[2](lang_feat)
        obj_feat_ff = self.obj_ff(obj_feat)
        obj_feat = obj_feat + obj_feat_ff
        obj_feat = self.obj_norms[2](obj_feat)

        return lang_feat, obj_feat


class Net(nn.Module):
    def __init__(self, args, obj_enc, word_enc, layers, obj_clf, lang_clf, obj_lang_clf, dropout=0.1):
        super().__init__()
        self.args = args
        self.dim = args.D
        self.depth = args.depth
        self.obj_enc = obj_enc
        self.word_enc = word_enc
        self.layers = nn.ModuleList(layers)

        # lang branches
        self.lang_clf_branch = MultiheadAttn(self.dim, 1, dropout=dropout)
        self.lang_match_branch = MultiheadAttn(self.dim, 1, dropout=dropout)

        # classificaiton heads
        self.obj_clf = obj_clf
        self.lang_clf = lang_clf
        self.obj_lang_clf = obj_lang_clf

        self.pos_fuse_layer = nn.Linear(self.dim + args.position_latent_dim, self.dim)

    def forward(self, batch: dict) -> dict:
        result = defaultdict(lambda: None)

        # prepare features
        obj_feat = get_siamese_features(self.obj_enc, batch['objects'],
                                        aggregator=torch.stack)
        position_feature = torch.squeeze(batch['positions'], -2)
        obj_feat = self.pos_fuse_layer(torch.cat((obj_feat, position_feature), -1))
        lang_feat = self.word_enc(batch['tokens'])

        # prepare masks
        word_mask = batch['word_mask']
        word_mask = word_mask.unsqueeze(1)
        word_mask_2d = word_mask * word_mask.transpose(1, 2)
        obj_mask = batch['object_mask']
        obj_mask = obj_mask.unsqueeze(1)
        obj_mask_2d = obj_mask * obj_mask.transpose(1, 2)
        co_mask_2d = word_mask.transpose(1, 2) * obj_mask

        # co-attention
        for layer in self.layers:
            lang_feat, obj_feat = layer(lang_feat, obj_feat, word_mask_2d, obj_mask_2d, co_mask_2d)

        # tasks
        lang_feat_clf = self.lang_clf_branch(lang_feat, lang_feat, lang_feat).max(dim=1)[0]
        lang_feat_match = self.lang_match_branch(lang_feat, lang_feat, lang_feat).max(dim=1)[0]

        # results
        result['class_logits'] = get_siamese_features(self.obj_clf, obj_feat, torch.stack)
        result['lang_logits'] = self.lang_clf(lang_feat_clf)
        lang_feat_expanded = lang_feat_match.unsqueeze(1).repeat(1, obj_feat.shape[1], 1)
        final_features = torch.cat((lang_feat_expanded, obj_feat), -1) 
        result['logits'] = get_siamese_features(self.obj_lang_clf, final_features, torch.cat)

        return result


class PositionalEncoding(nn.Module):
    """
        PE(pos, 2i)=sin(pos/(10000^(2*i/dim)))
        PE(pos, 2i+1)=cos(pos/(10000^(2*i/dim)))
    """
    def __init__(self, dim, max_len=100, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: B x nword x D
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)



def instantiate_net(args: argparse.Namespace, vocab: Vocabulary, n_obj_classes: int) -> nn.Module:
    dim = args.D
    obj_enc = single_object_encoder(dim)
    word_enc = nn.Sequential(
        nn.Embedding(len(vocab), dim, padding_idx=vocab.pad),
        nn.Dropout(args.dropout),
        nn.Linear(dim, dim),
        PositionalEncoding(dim)
    )
    layers = [instantiate_coattentionnet(args) for _ in range(args.depth)]
    obj_clf = MLP(dim, [128, 256, n_obj_classes], dropout_rate=0.15)
    lang_clf = MLP(dim, [128, 256, n_obj_classes], dropout_rate=0.2)
    object_language_clf = MLP(dim * 2, out_channels=[128, 64, 1], dropout_rate=0.05)
    return Net(args, obj_enc, word_enc, layers, obj_clf, lang_clf, object_language_clf, dropout=args.dropout)
