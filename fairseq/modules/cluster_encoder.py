# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional

import torch
import torch.nn as nn
from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.quant_noise import quant_noise
from fairseq.modules.fairseq_dropout import FairseqDropout
from torch import Tensor


class TransformerClusterEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args):
        super().__init__()
        self.encoders = []
        for i in range(0, 4):

            self.encoders.append({})
            self.encoders[i]['embed_dim'] = args.encoder_embed_dim
            self.encoders[i]['quant_noise'] = getattr(args, "quant_noise_pq", 0)
            self.encoders[i]['quant_noise_block_size'] = getattr(args, "quant_noise_pq_block_size", 8)

            self.encoders[i]['self_attn'] = self.build_self_attention(self.encoders[i]['embed_dim'], i, args)
            self.encoders[i]['self_attn_layer_norm'] = LayerNorm(self.encoders[i]['embed_dim'])
            self.encoders[i]['dropout_module'] = FairseqDropout(args.dropout, module_name=self.__class__.__name__)
            self.encoders[i]['activation_fn'] = utils.get_activation_fn(
                activation=getattr(args, "activation_fn", "relu")
            )
            activation_dropout_p = getattr(args, "activation_dropout", 0)
            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = getattr(args, "relu_dropout", 0)
            self.encoders[i]['activation_dropout_module'] = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
            self.encoders[i]['normalize_before'] = args.encoder_normalize_before
            self.encoders[i]['fc1'] = self.build_fc1(
                self.encoders[i]['embed_dim'], args.encoder_ffn_embed_dim, self.encoders[i]['quant_noise'], self.encoders[i]['quant_noise_block_size']
            )
            self.encoders[i]['fc2'] = self.build_fc2(
                args.encoder_ffn_embed_dim, self.encoders[i]['embed_dim'], self.encoders[i]['quant_noise'], self.encoders[i]['quant_noise_block_size']
            )

            self.encoders[i]['final_layer_norm'] = LayerNorm(self.encoders[i]['embed_dim'])




    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size)

    def build_self_attention(self, embed_dim, index, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.encoders[index]['quant_noise'],
            qn_block_size=self.encoders[index]['quant_noise_block_size'],
        )

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(self, x, encoder_padding_mask, index, attn_mask: Optional[Tensor] = None):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        

        #print("cluster forward")

        # print(self.encoders[0]['self_attn'].k_proj.weight)

        # print(self.encoders[1]['self_attn'].k_proj.weight)

        # print(self.encoders[2]['self_attn'].k_proj.weight)

        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.encoders[index]['normalize_before']:
            x = self.encoders[index]['self_attn_layer_norm'](x)
        x, _ = self.encoders[index]['self_attn'](
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            attn_mask=attn_mask,
        )

        x = self.encoders[index]['dropout_module'](x)
        x = residual + x
        if not self.encoders[index]['normalize_before']:
            x = self.encoders[index]['self_attn_layer_norm'](x)

        residual = x
        if self.encoders[index]['normalize_before']:
            x = self.encoders[index]['final_layer_norm'](x)

        x = self.encoders[index]['activation_fn'](self.encoders[index]['fc1'](x))
        x = self.encoders[index]['activation_dropout_module'](x)
        x = self.encoders[index]['fc2'](x)
        x = self.encoders[index]['dropout_module'](x)
        x = residual + x
        if not self.encoders[index]['normalize_before']:
            x = self.encoders[index]['final_layer_norm'](x)
        return x
