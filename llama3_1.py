from typing import Optional, Tuple
from dataclasses import dataclass

import math

import torch
import torch.nn as nn


@dataclass
class ModelArgs:
    vocab_size: int = -1
    padding_idx: int = -1
    dim: int = 12
    num_layers: int = 1
    num_heads: int = 6
    num_kv_heads: Optional[int] = None
    rms_norm_eps: float = 1e-5



class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    

class Llama3_1(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.freqs_cis = precompute_freqs_cis() # TODO

        self.embeddings = nn.Embedding(args.vocab_size, args.dim, padding_idx=)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(layer_id, args)) # TODO

        self.norm = RMSNorm(args.dim, args.rms_norm_eps)
        self.lm_head = nn.Linear(in_features=args.dim, out_features=args.vocab_size, bias=False)

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int) -> torch.Tensor:
        B, L = tokens.shape
        embedds = self.embeddings(tokens)

        freqs_cis = # TODO

        mask = None
        if L > 1:
            mask = torch.full((L, L), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((L, start_pos), device=tokens.device), mask]).type_as(h)

        for layer in self.layers:
            h = layer(h, start_pos, freqs_cis, mask)

        h = self.norm(h)
        return self.lm_head(h).float()
