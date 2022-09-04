# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from ms_cm.bert_layers import *

class Transformer(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        # TransformerEncoderLayerThin
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # TransformerDecoderLayerThin
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed = pos_embed.permute(1, 0, 2)   # (L, batch_size, d)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, batch_size, d)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)  # (#layers, #queries, batch_size, d)
        hs = hs.transpose(1, 2)  # (#layers, batch_size, #qeries, d)
        # memory = memory.permute(1, 2, 0)  # (batch_size, d, L)
        memory = memory.transpose(0, 1)  # (batch_size, L, d)
        return hs, memory



class Transformer_justencoder(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()

        # TransformerEncoderLayerThin
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        # # TransformerDecoderLayerThin
        # decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
        #                                         dropout, activation, normalize_before)
        # decoder_norm = nn.LayerNorm(d_model)
        # self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
        #                                   return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC
        bs, l, d = src.shape
        src = src.permute(1, 0, 2)  # (L, batch_size, d)
        pos_embed = pos_embed.permute(1, 0, 2)   # (L, batch_size, d)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)  # (#queries, batch_size, d)

        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)  # (L, batch_size, d)
        # hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
        #                   pos=pos_embed, query_pos=query_embed)  # (#layers, #queries, batch_size, d)
        # hs = hs.transpose(1, 2)  # (#layers, batch_size, #qeries, d)
        # memory = memory.permute(1, 2, 0)  # (batch_size, d, L)
        memory = memory.transpose(0, 1)  # (batch_size, L, d)
        return  memory


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        intermediate = []

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
            if self.return_intermediate:
                intermediate.append(output)

        if self.norm is not None:
            output = self.norm(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayerThin(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src2 = self.linear(src2)
        src = src + self.dropout(src2)
        src = self.norm(src)
        # src = src + self.dropout1(src2)
        # src = self.norm1(src)
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        # src = src + self.dropout2(src2)
        # src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        """not used"""
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class TransformerDecoderLayerThin(nn.Module):
    """removed intermediate layer"""
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, d_model)
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        # self.dropout = nn.Dropout(dropout)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        # self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        # self.dropout3 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt2 = self.linear1(tgt2)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        # tgt = tgt + self.dropout2(tgt2)
        # tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        # tgt = tgt + self.dropout3(tgt2)
        # tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)



def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def build_transformer(args):
    return Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


def build_transformer_encoder(args):
    return Transformer_justencoder(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        normalize_before=args.pre_norm
    )

def build_cross_self_encoder(args):
    return Cross_self_attention(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_cross_encoder_layers=args.num_cross_encoder_layers,
        normalize_before=args.pre_norm,
        cross_first=args.cross_first,
        input_dropout=args.cross_input_dropout,
        cross_text_use_ori=args.cross_text_use_ori,
        args=args
    )

def build_cross_self_encoder_fixcross(args):
    return Cross_self_attention_fixcross(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_cross_encoder_layers=args.num_cross_encoder_layers,
        normalize_before=args.pre_norm,
        cross_first=args.cross_first,
        input_dropout=args.cross_input_dropout,
        cross_text_use_ori=args.cross_text_use_ori,
        args=args
    )

def build_cross_self_encoder_fixcross_addonlyvtrans(args):
    return Cross_self_attention_fixcross_addonlyvtrans(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_onlyv_layers=args.onlyv_layers,
        num_encoder_layers=args.enc_layers,
        num_cross_encoder_layers=args.num_cross_encoder_layers,
        normalize_before=args.pre_norm,
        cross_first=args.cross_first,
        input_dropout=args.cross_input_dropout,
        cross_text_use_ori=args.cross_text_use_ori,
        args=args
    )


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


# class CrossAttention(nn.Module):
#     def __init__(self, d_model, num_attention_heads, dropout_prob):
#         super(CrossAttention, self).__init__()
#         if d_model % num_attention_heads != 0:
#             raise ValueError(
#                 "The hidden size (%d) is not a multiple of the number of attention "
#                 "heads (%d)" % (d_model, num_attention_heads)
#             )
#         self.num_attention_heads = num_attention_heads
#         self.attention_head_size = int(d_model / num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
#         # self.scale_act_fn = ACT2FN['relu']

#         self.query_vid = nn.Linear(d_model, self.all_head_size)
#         self.key_text = nn.Linear(d_model, self.all_head_size)
#         self.value_text = nn.Linear(d_model, self.all_head_size)
#         # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

#         self.dropout1 = nn.Dropout(dropout_prob)


#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (
#             self.num_attention_heads,
#             self.attention_head_size,
#         )
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(self, input_tensor_vid, attention_mask_vid, input_tensor_text, attention_mask_text):

#         # query: vid. key and value: text
#         mixed_query_layer1 = self.query_vid(input_tensor_vid)
#         mixed_key_layer1 = self.key_text(input_tensor_text)
#         mixed_value_layer1 = self.value_text(input_tensor_text)
#         # mixed_logit_layer1 = self.logit1(input_tensor1)

#         query_layer1 = self.transpose_for_scores(mixed_query_layer1)
#         key_layer1 = self.transpose_for_scores(mixed_key_layer1)
#         value_layer1 = self.transpose_for_scores(mixed_value_layer1)
#         # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

#         # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
#         attention_scores1 = torch.matmul(query_layer1, key_layer1.transpose(-1, -2))
#         attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
#         attention_scores1 = attention_scores1 + attention_mask_vid

#         # Normalize the attention scores to probabilities.
#         attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs1 = self.dropout1(attention_probs1)

#         context_layer1 = torch.matmul(attention_probs1, value_layer1)
#         context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
#         context_layer1 = context_layer1.view(*new_context_layer_shape1)

#         return context_layer1, attention_probs1

# class Cross_Output(nn.Module):
#     def __init__(self, d_model,dim_feedforward,dropout_prob, activation):
#         super(Cross_Output, self).__init__()

#         # self.dense1 = nn.Linear(hidden_size, hidden_size)
#         # self.LayerNorm1 = BertLayerNorm(hidden_size, eps=1e-12)
#         # self.dropout1 = nn.Dropout(dropout_prob)

#         self.linear1 = nn.Linear(d_model, dim_feedforward)
#         self.dropout = nn.Dropout(dropout_prob)
#         self.linear2 = nn.Linear(dim_feedforward, d_model)

#         self.norm1 = nn.LayerNorm(d_model)
#         self.norm2 = nn.LayerNorm(d_model)
#         self.dropout1 = nn.Dropout(dropout_prob)
#         self.dropout2 = nn.Dropout(dropout_prob)

#         self.activation = _get_activation_fn(activation)


#     def forward(self, src2, src):

#         src = src + self.dropout1(src2)
#         src = self.norm1(src)
#         src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
#         src = src + self.dropout2(src2)
#         src = self.norm2(src)

#         return hidden_states1


# class Cross_layer(nn.Module):
#     def __init__(self, d_model, dim_feedforward, num_attention_heads, dropout_prob, activation):
#         super(Cross_layer,self).__init__()
#         self.cross_attn = CrossAttention(d_model, num_attention_heads, dropout_prob)
#         self.out_layer = Cross_Output(d_model, dim_feedforward, dropout_prob, activation)


#     def forward(self,input_tensor_vid, attention_mask_vid, input_tensor_text, attention_mask_text):
#         cross_output, cross_attention_prob = self.cross_attn(input_tensor_vid, attention_mask_vid, input_tensor_text, attention_mask_text)
#         cross_output_final = self.out_layer(cross_output,input_tensor_vid)

#         return cross_output_final


class CrossEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, vid_embedding, vid_attention_mask, txt_embedding, 
                        txt_attention_mask, co_attention_mask,use_co_attention_mask):

        for layer in self.layers:
            vid_embedding, txt_embedding, co_attention_probs  = layer(vid_embedding, vid_attention_mask, txt_embedding, 
                    txt_attention_mask, co_attention_mask,use_co_attention_mask)

        return vid_embedding, txt_embedding



class Cross_self_attention(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_cross_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", cross_first=False,input_dropout=0.1, normalize_before=False,cross_text_use_ori=False,args=None):
        super().__init__()

        self.cross_first = cross_first
        self.cross_text_use_ori = cross_text_use_ori
        self.num_encoder_layers = num_encoder_layers
        # TransformerEncoderLayerThin
        cross_encoder_layer = BertConnectionLayer(args)
        self.cross_encoder = CrossEncoder(cross_encoder_layer, num_cross_encoder_layers)

        if num_encoder_layers != 0:

            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        elif num_encoder_layers == 0:
            assert cross_first
            pass 


        if cross_first:
            self.cross_vid_LayerNorm = BertLayerNorm(d_model, eps=1e-12)
            self.cross_vid_dropout = nn.Dropout(input_dropout)

            self.cross_txt_LayerNorm = BertLayerNorm(d_model, eps=1e-12)
            self.cross_txt_dropout = nn.Dropout(input_dropout)




        self._reset_parameters()
        # self.apply(self.init_bert_weights)

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed):
        if self.cross_first:
            return self.forward_cross_first(src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed)
        else:
            return self.forward_self_first(src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed)

    def forward_cross_first(self, src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC

        use_co_attention_mask = False
        co_attention_mask = None


        mask = torch.cat([mask_vid, mask_txt], dim=1).bool()  # (bsz, L_vid+L_txt)
        transformer_encoder_mask = ~mask
        transformer_encoder_pos = torch.cat([vid_pos_embed, txt_pos_embed], dim=1)
        # transformer_encoder_src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)

        vid_embedding = self.cross_vid_LayerNorm(src_vid + vid_pos_embed)
        vid_embedding = self.cross_vid_dropout(vid_embedding)

        txt_embedding = self.cross_txt_LayerNorm(src_txt + txt_pos_embed)
        txt_embedding = self.cross_txt_dropout(txt_embedding)

        extended_mask_txt = mask_txt.unsqueeze(1).unsqueeze(2)
        extended_mask_vid = mask_vid.unsqueeze(1).unsqueeze(2)

        extended_mask_txt = extended_mask_txt.to(dtype=torch.float32)
        extended_mask_txt = (1.0 - extended_mask_txt) * -10000.0

        extended_mask_vid = extended_mask_vid.to(dtype=torch.float32)
        extended_mask_vid = (1.0 - extended_mask_vid) * -10000.0

        # print("vid_embedding.shape: {}".format(vid_embedding.shape))
        # print("txt_embedding.shape: {}".format(txt_embedding.shape))
        cross_vid_output, cross_txt_output = self.cross_encoder(vid_embedding, extended_mask_vid, txt_embedding, 
                            extended_mask_txt, co_attention_mask,use_co_attention_mask)  # (L, batch_size, d)
        
        # print('cross_vid_output.shape: {}'.format(cross_vid_output.shape))
        # print("cross_txt_output.shape: {}".format(cross_txt_output.shape))
        # exit()

        if self.num_encoder_layers==0:
            return cross_vid_output, cross_txt_output
        transformer_encoder_src = torch.cat([cross_vid_output, cross_txt_output], dim=1)  # (bsz, L_vid+L_txt, d)
        
        transformer_encoder_src = transformer_encoder_src.permute(1, 0, 2)  # (L, batch_size, d)
        transformer_encoder_pos = transformer_encoder_pos.permute(1, 0, 2)   # (L, batch_size, d)

        transformer_encoder_output = self.encoder(transformer_encoder_src, src_key_padding_mask=transformer_encoder_mask, pos=transformer_encoder_pos)


        transformer_encoder_output = transformer_encoder_output.transpose(0, 1)  # (batch_size, L, d)\

        txt_output = transformer_encoder_output[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_output = transformer_encoder_output[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        return  vid_output, txt_output


    def forward_self_first(self, src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC

        use_co_attention_mask = False
        co_attention_mask = None


        mask = torch.cat([mask_vid, mask_txt], dim=1).bool()  # (bsz, L_vid+L_txt)
        transformer_encoder_mask = ~mask
        transformer_encoder_pos = torch.cat([vid_pos_embed, txt_pos_embed], dim=1)
        # transformer_encoder_src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)

        transformer_encoder_src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        
        transformer_encoder_src = transformer_encoder_src.permute(1, 0, 2)  # (L, batch_size, d)
        transformer_encoder_pos = transformer_encoder_pos.permute(1, 0, 2)   # (L, batch_size, d)

        transformer_encoder_output = self.encoder(transformer_encoder_src, src_key_padding_mask=transformer_encoder_mask, pos=transformer_encoder_pos)


        transformer_encoder_output = transformer_encoder_output.transpose(0, 1)  # (batch_size, L, d)

        transformer_encoder_output_txt = transformer_encoder_output[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        transformer_encoder_output_vid = transformer_encoder_output[:, :src_vid.shape[1]]  # (bsz, L_vid, d)

        # vid_embedding = self.cross_vid_LayerNorm(transformer_encoder_output_vid + vid_pos_embed)
        # vid_embedding = self.cross_vid_dropout(vid_embedding)

        # txt_embedding = self.cross_txt_LayerNorm(transformer_encoder_output_txt + txt_pos_embed)
        # txt_embedding = self.cross_txt_dropout(txt_embedding)

        extended_mask_txt = mask_txt.unsqueeze(1).unsqueeze(2)
        extended_mask_vid = mask_vid.unsqueeze(1).unsqueeze(2)

        extended_mask_txt = extended_mask_txt.to(dtype=torch.float32)
        extended_mask_txt = (1.0 - extended_mask_txt) * -10000.0

        extended_mask_vid = extended_mask_vid.to(dtype=torch.float32)
        extended_mask_vid = (1.0 - extended_mask_vid) * -10000.0

        # print("vid_embedding.shape: {}".format(transformer_encoder_output_vid.shape))
        # print("txt_embedding.shape: {}".format(transformer_encoder_output_txt.shape))
        if not self.cross_text_use_ori:
            cross_vid_output, cross_txt_output = self.cross_encoder(transformer_encoder_output_vid, extended_mask_vid, transformer_encoder_output_txt, 
                                extended_mask_txt, co_attention_mask, use_co_attention_mask)  # (L, batch_size, d)
        else:
            cross_vid_output, cross_txt_output = self.cross_encoder(transformer_encoder_output_vid, extended_mask_vid, src_txt, 
                                extended_mask_txt, co_attention_mask, use_co_attention_mask)  # (L, batch_size, d)

        # print('cross_vid_output.shape: {}'.format(cross_vid_output.shape))
        # print("cross_txt_output.shape: {}".format(cross_txt_output.shape))
        # exit()

        vid_output = cross_vid_output
        txt_output = cross_txt_output
        return  vid_output, txt_output


class Cross_self_attention_fixcross(nn.Module):

    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_cross_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", cross_first=False,input_dropout=0.1, normalize_before=False,cross_text_use_ori=False,args=None):
        super().__init__()

        self.cross_first = cross_first
        self.cross_text_use_ori = cross_text_use_ori
        self.num_encoder_layers = num_encoder_layers
        # TransformerEncoderLayerThin
        cross_encoder_layer = BertConnectionLayer(args)
        self.cross_encoder = CrossEncoder(cross_encoder_layer, num_cross_encoder_layers)

        if num_encoder_layers != 0:

            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        elif num_encoder_layers == 0:
            assert cross_first
            pass 

        self._reset_parameters()
        # self.apply(self.init_bert_weights)

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed):
        if self.cross_first:
            return self.forward_cross_first(src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed)
        else:
            return self.forward_self_first(src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed)

    def forward_cross_first(self, src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC

        use_co_attention_mask = False
        co_attention_mask = None


        mask = torch.cat([mask_vid, mask_txt], dim=1).bool()  # (bsz, L_vid+L_txt)
        transformer_encoder_mask = ~mask
        transformer_encoder_pos = torch.cat([vid_pos_embed, txt_pos_embed], dim=1)

        vid_embedding = src_vid + vid_pos_embed

        txt_embedding = src_txt + txt_pos_embed

        extended_mask_txt = mask_txt.unsqueeze(1).unsqueeze(2)
        extended_mask_vid = mask_vid.unsqueeze(1).unsqueeze(2)

        extended_mask_txt = extended_mask_txt.to(dtype=torch.float32)
        extended_mask_txt = (1.0 - extended_mask_txt) * -10000.0

        extended_mask_vid = extended_mask_vid.to(dtype=torch.float32)
        extended_mask_vid = (1.0 - extended_mask_vid) * -10000.0

        # print("vid_embedding.shape: {}".format(vid_embedding.shape))
        # print("txt_embedding.shape: {}".format(txt_embedding.shape))
        cross_vid_output, cross_txt_output = self.cross_encoder(vid_embedding, extended_mask_vid, txt_embedding, 
                            extended_mask_txt, co_attention_mask,use_co_attention_mask)  # (L, batch_size, d)
        
        # print('cross_vid_output.shape: {}'.format(cross_vid_output.shape))
        # print("cross_txt_output.shape: {}".format(cross_txt_output.shape))
        # exit()

        if self.num_encoder_layers==0:
            return cross_vid_output, cross_txt_output
        transformer_encoder_src = torch.cat([cross_vid_output, cross_txt_output], dim=1)  # (bsz, L_vid+L_txt, d)
        
        transformer_encoder_src = transformer_encoder_src.permute(1, 0, 2)  # (L, batch_size, d)
        transformer_encoder_pos = transformer_encoder_pos.permute(1, 0, 2)   # (L, batch_size, d)

        transformer_encoder_output = self.encoder(transformer_encoder_src, src_key_padding_mask=transformer_encoder_mask, pos=transformer_encoder_pos)


        transformer_encoder_output = transformer_encoder_output.transpose(0, 1)  # (batch_size, L, d)\

        txt_output = transformer_encoder_output[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_output = transformer_encoder_output[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        return  vid_output, txt_output


    def forward_self_first(self, src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC

        use_co_attention_mask = False
        co_attention_mask = None


        mask = torch.cat([mask_vid, mask_txt], dim=1).bool()  # (bsz, L_vid+L_txt)
        transformer_encoder_mask = ~mask
        transformer_encoder_pos = torch.cat([vid_pos_embed, txt_pos_embed], dim=1)
        # transformer_encoder_src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)

        transformer_encoder_src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        
        transformer_encoder_src = transformer_encoder_src.permute(1, 0, 2)  # (L, batch_size, d)
        transformer_encoder_pos = transformer_encoder_pos.permute(1, 0, 2)   # (L, batch_size, d)

        transformer_encoder_output = self.encoder(transformer_encoder_src, src_key_padding_mask=transformer_encoder_mask, pos=transformer_encoder_pos)


        transformer_encoder_output = transformer_encoder_output.transpose(0, 1)  # (batch_size, L, d)

        transformer_encoder_output_txt = transformer_encoder_output[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        transformer_encoder_output_vid = transformer_encoder_output[:, :src_vid.shape[1]]  # (bsz, L_vid, d)

        # vid_embedding = self.cross_vid_LayerNorm(transformer_encoder_output_vid + vid_pos_embed)
        # vid_embedding = self.cross_vid_dropout(vid_embedding)

        # txt_embedding = self.cross_txt_LayerNorm(transformer_encoder_output_txt + txt_pos_embed)
        # txt_embedding = self.cross_txt_dropout(txt_embedding)

        extended_mask_txt = mask_txt.unsqueeze(1).unsqueeze(2)
        extended_mask_vid = mask_vid.unsqueeze(1).unsqueeze(2)

        extended_mask_txt = extended_mask_txt.to(dtype=torch.float32)
        extended_mask_txt = (1.0 - extended_mask_txt) * -10000.0

        extended_mask_vid = extended_mask_vid.to(dtype=torch.float32)
        extended_mask_vid = (1.0 - extended_mask_vid) * -10000.0

        # print("vid_embedding.shape: {}".format(transformer_encoder_output_vid.shape))
        # print("txt_embedding.shape: {}".format(transformer_encoder_output_txt.shape))
        if not self.cross_text_use_ori:
            cross_vid_output, cross_txt_output = self.cross_encoder(transformer_encoder_output_vid, extended_mask_vid, transformer_encoder_output_txt, 
                                extended_mask_txt, co_attention_mask, use_co_attention_mask)  # (L, batch_size, d)
        else:
            cross_vid_output, cross_txt_output = self.cross_encoder(transformer_encoder_output_vid, extended_mask_vid, src_txt, 
                                extended_mask_txt, co_attention_mask, use_co_attention_mask)  # (L, batch_size, d)

        # print('cross_vid_output.shape: {}'.format(cross_vid_output.shape))
        # print("cross_txt_output.shape: {}".format(cross_txt_output.shape))
        # exit()

        vid_output = cross_vid_output
        txt_output = cross_txt_output
        return  vid_output, txt_output


class Onlyvtrans_encoder(nn.Module):
    def __init__(self, d_model=512, nhead=8,num_onlyv_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", cross_first=False,input_dropout=0.1, normalize_before=False,args=None):
        super().__init__()
        onlyv_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.onlyv_encoder = TransformerEncoder(onlyv_encoder_layer, num_onlyv_layers, encoder_norm)


        self._reset_parameters()
        # self.apply(self.init_bert_weights)

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, src_vid, mask_vid, vid_pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC

        onlyv_mask = mask_vid.bool()  # (bsz, L_vid+L_txt)
        onlyv_encoder_mask = ~onlyv_mask
        onlyv_encoder_pos = vid_pos_embed

        
        only_vid_embedding = src_vid.permute(1, 0, 2)  # (L, batch_size, d)
        onlyv_encoder_pos = onlyv_encoder_pos.permute(1, 0, 2)   # (L, batch_size, d)

        only_vid_embedding = self.onlyv_encoder(only_vid_embedding, src_key_padding_mask=onlyv_encoder_mask, pos=onlyv_encoder_pos)


        only_vid_embedding = only_vid_embedding.transpose(0, 1)  # (batch_size, L, d)

        return only_vid_embedding






class Cross_self_attention_fixcross_addonlyvtrans(nn.Module):

    def __init__(self, d_model=512, nhead=8,num_onlyv_layers=6, num_encoder_layers=6, num_cross_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", cross_first=False,input_dropout=0.1, normalize_before=False,cross_text_use_ori=False,args=None):
        super().__init__()

        self.cross_first = cross_first
        self.cross_text_use_ori = cross_text_use_ori
        self.num_encoder_layers = num_encoder_layers
        self.num_onlyv_layers = num_onlyv_layers
        # TransformerEncoderLayerThin
        cross_encoder_layer = BertConnectionLayer(args)
        self.cross_encoder = CrossEncoder(cross_encoder_layer, num_cross_encoder_layers)


        if num_onlyv_layers != 0:
            onlyv_encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.onlyv_encoder = TransformerEncoder(onlyv_encoder_layer, num_onlyv_layers, encoder_norm)


        if num_encoder_layers != 0:

            encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                    dropout, activation, normalize_before)
            encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
            self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        elif num_encoder_layers == 0:
            assert cross_first
            pass 

        self._reset_parameters()
        # self.apply(self.init_bert_weights)

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def forward(self, src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed):
        if self.cross_first:
            return self.forward_cross_first(src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed)
        else:
            return self.forward_self_first(src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed)

    def forward_cross_first(self, src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC

        use_co_attention_mask = False
        co_attention_mask = None


        mask = torch.cat([mask_vid, mask_txt], dim=1).bool()  # (bsz, L_vid+L_txt)
        transformer_encoder_mask = ~mask
        transformer_encoder_pos = torch.cat([vid_pos_embed, txt_pos_embed], dim=1)


        if self.num_onlyv_layers!=0:
            onlyv_mask = mask_vid.bool()  # (bsz, L_vid+L_txt)
            onlyv_encoder_mask = ~only_v_mask
            onlyv_encoder_pos = vid_pos_embed
            # transformer_encoder_src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)

            transformer_encoder_src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
            
            src_vid = src_vid.permute(1, 0, 2)  # (L, batch_size, d)
            onlyv_encoder_pos = onlyv_encoder_pos.permute(1, 0, 2)   # (L, batch_size, d)

            src_vid = self.encoder(src_vid, src_key_padding_mask=onlyv_encoder_mask, pos=onlyv_encoder_pos)


            src_vid = src_vid.transpose(0, 1)  # (batch_size, L, d)

        onlyv_embedding = src_vid

        vid_embedding = src_vid + vid_pos_embed

        txt_embedding = src_txt + txt_pos_embed

        extended_mask_txt = mask_txt.unsqueeze(1).unsqueeze(2)
        extended_mask_vid = mask_vid.unsqueeze(1).unsqueeze(2)

        extended_mask_txt = extended_mask_txt.to(dtype=torch.float32)
        extended_mask_txt = (1.0 - extended_mask_txt) * -10000.0

        extended_mask_vid = extended_mask_vid.to(dtype=torch.float32)
        extended_mask_vid = (1.0 - extended_mask_vid) * -10000.0

        # print("vid_embedding.shape: {}".format(vid_embedding.shape))
        # print("txt_embedding.shape: {}".format(txt_embedding.shape))
        cross_vid_output, cross_txt_output = self.cross_encoder(vid_embedding, extended_mask_vid, txt_embedding, 
                            extended_mask_txt, co_attention_mask,use_co_attention_mask)  # (L, batch_size, d)
        
        # print('cross_vid_output.shape: {}'.format(cross_vid_output.shape))
        # print("cross_txt_output.shape: {}".format(cross_txt_output.shape))
        # exit()

        if self.num_encoder_layers==0:
            return cross_vid_output, cross_txt_output
        transformer_encoder_src = torch.cat([cross_vid_output, cross_txt_output], dim=1)  # (bsz, L_vid+L_txt, d)
        
        transformer_encoder_src = transformer_encoder_src.permute(1, 0, 2)  # (L, batch_size, d)
        transformer_encoder_pos = transformer_encoder_pos.permute(1, 0, 2)   # (L, batch_size, d)

        transformer_encoder_output = self.encoder(transformer_encoder_src, src_key_padding_mask=transformer_encoder_mask, pos=transformer_encoder_pos)


        transformer_encoder_output = transformer_encoder_output.transpose(0, 1)  # (batch_size, L, d)\

        txt_output = transformer_encoder_output[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_output = transformer_encoder_output[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        return  onlyv_embedding, vid_output, txt_output


    def forward_self_first(self, src_vid, mask_vid, src_txt, mask_txt, vid_pos_embed, txt_pos_embed):
        """
        Args:
            src: (batch_size, L, d)
            mask: (batch_size, L)
            query_embed: (#queries, d)
            pos_embed: (batch_size, L, d) the same as src

        Returns:

        """
        # flatten NxCxHxW to HWxNxC

        use_co_attention_mask = False
        co_attention_mask = None


        mask = torch.cat([mask_vid, mask_txt], dim=1).bool()  # (bsz, L_vid+L_txt)
        transformer_encoder_mask = ~mask
        transformer_encoder_pos = torch.cat([vid_pos_embed, txt_pos_embed], dim=1)
        # transformer_encoder_src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)

        transformer_encoder_src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        
        transformer_encoder_src = transformer_encoder_src.permute(1, 0, 2)  # (L, batch_size, d)
        transformer_encoder_pos = transformer_encoder_pos.permute(1, 0, 2)   # (L, batch_size, d)

        transformer_encoder_output = self.encoder(transformer_encoder_src, src_key_padding_mask=transformer_encoder_mask, pos=transformer_encoder_pos)


        transformer_encoder_output = transformer_encoder_output.transpose(0, 1)  # (batch_size, L, d)

        transformer_encoder_output_txt = transformer_encoder_output[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        transformer_encoder_output_vid = transformer_encoder_output[:, :src_vid.shape[1]]  # (bsz, L_vid, d)

        # vid_embedding = self.cross_vid_LayerNorm(transformer_encoder_output_vid + vid_pos_embed)
        # vid_embedding = self.cross_vid_dropout(vid_embedding)

        # txt_embedding = self.cross_txt_LayerNorm(transformer_encoder_output_txt + txt_pos_embed)
        # txt_embedding = self.cross_txt_dropout(txt_embedding)

        extended_mask_txt = mask_txt.unsqueeze(1).unsqueeze(2)
        extended_mask_vid = mask_vid.unsqueeze(1).unsqueeze(2)

        extended_mask_txt = extended_mask_txt.to(dtype=torch.float32)
        extended_mask_txt = (1.0 - extended_mask_txt) * -10000.0

        extended_mask_vid = extended_mask_vid.to(dtype=torch.float32)
        extended_mask_vid = (1.0 - extended_mask_vid) * -10000.0

        # print("vid_embedding.shape: {}".format(transformer_encoder_output_vid.shape))
        # print("txt_embedding.shape: {}".format(transformer_encoder_output_txt.shape))
        if not self.cross_text_use_ori:
            cross_vid_output, cross_txt_output = self.cross_encoder(transformer_encoder_output_vid, extended_mask_vid, transformer_encoder_output_txt, 
                                extended_mask_txt, co_attention_mask, use_co_attention_mask)  # (L, batch_size, d)
        else:
            cross_vid_output, cross_txt_output = self.cross_encoder(transformer_encoder_output_vid, extended_mask_vid, src_txt, 
                                extended_mask_txt, co_attention_mask, use_co_attention_mask)  # (L, batch_size, d)

        # print('cross_vid_output.shape: {}'.format(cross_vid_output.shape))
        # print("cross_txt_output.shape: {}".format(cross_txt_output.shape))
        # exit()

        vid_output = cross_vid_output
        txt_output = cross_txt_output
        return  vid_output, txt_output
