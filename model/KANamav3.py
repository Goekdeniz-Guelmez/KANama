from typing import Optional
from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kan import KANLinear
from model.utils import RMSNorm, precompute_freqs_cis, apply_rotary_emb, repeat_kv


@dataclass
class ModelArgs:
    vocab_size: int = -1
    pad_id: int = -1
    eos_id: int = pad_id

    dim: int = 32
    n_layers: int = 8

    n_heads: int = 8
    n_kv_heads: Optional[int] = None

    use_kan: bool = True
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None

    rms_norm_eps: float = 1e-5

    rope_theta: float = 500000
    use_scaled_rope: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        self.n_heads = args.n_heads

        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = args.dim // self.n_heads

        self.q_proj = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)

        self.softmax_temp_proj = nn.Linear(args.dim, 1, bias=False)  # Add softmax temperature projection\
        self.softmax_temp_act = F.silu
        self.current_softmax_temp = None

        self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
        self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))

        self.out_proj = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        xq = xq.view(bsz, seqlen, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        self.cache_k = self.cache_k.to(xq)
        self.cache_v = self.cache_v.to(xq)

        if self.training:
            # Detach cache before updating during training
            self.cache_k = self.cache_k.detach()
            self.cache_v = self.cache_v.detach()

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        queries = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(queries, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)

        # Apply softmax temperature projection
        self.current_softmax_temp = self.softmax_temp_act(self.softmax_temp_proj(x))
        self.current_softmax_temp = torch.clamp(self.current_softmax_temp, min=0.1, max=10.0).mean().item() + 1e-6 # clamp the temperature and ensure temp is positive

        scores = F.softmax(scores.float() * self.current_softmax_temp, dim=-1).type_as(queries)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs, hidden_dim: int):
        super().__init__()
        self.multiple_of = args.multiple_of
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * hidden_dim)
        hidden_dim = self.multiple_of * ((hidden_dim + self.multiple_of - 1) // self.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, args.dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))
    

class KANMLP(nn.Module):
    def __init__(self, args: ModelArgs, hidden_dim: int):
        super().__init__()
        self.w1 = KANLinear(args.dim, hidden_dim)
        self.w3 = KANLinear(args.dim, hidden_dim)
        self.w2 = KANLinear(hidden_dim, args.dim, base_activation=nn.Identity, enable_standalone_scale_spline=True)

    def forward(self, x):
        return self.w2(self.w1(x) * self.w3(x))
        

class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.layer_id = layer_id

        self.attention_norm = RMSNorm(args.dim, args.rms_norm_eps)
        self.attention = Attention(args)

        self.mlp_norm = RMSNorm(args.dim, args.rms_norm_eps)

        if args.use_kan:
            self.mlp = KANMLP(args, 4 * args.dim)
        else:
            self.mlp = MLP(args, 4 * args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.mlp(self.mlp_norm(h))
        return out



class KANamav3(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta, args.use_scaled_rope)

        self.embeddings = nn.Embedding(args.vocab_size, args.dim, padding_idx=args.pad_id)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.dim, args.rms_norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor, start_pos: int = 0, targets: Optional[int] = None) -> torch.Tensor:
        B, L = tokens.shape
        embedds = self.embeddings(tokens)

        self.freqs_cis = self.freqs_cis.to(embedds.device)
        freqs_cis = self.freqs_cis[start_pos : start_pos + L]

        mask = None
        if L > 1:
            mask = torch.full((L, L), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((L, start_pos), device=tokens.device), mask]).type_as(embedds)

        for layer in self.layers:
            h = layer(embedds, start_pos, freqs_cis, mask)

        h = self.norm(h)

        logits = self.lm_head(h).float()

        loss = None
        if targets is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.args.vocab_size)
            shift_targets = shift_targets.view(-1)
            # Enable model parallelism
            shift_targets = shift_targets.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_targets)

        return logits, loss