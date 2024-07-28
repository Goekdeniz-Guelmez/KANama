from typing import Optional, Tuple
from dataclasses import dataclass

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kan import KANLinear


@dataclass
class ModelArgs:
    vocab_size: int = 20
    padding_idx: int = 1

    dim: int = 12
    n_layers: int = 1

    n_heads: int = 6
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

def apply_scaling(freqs: torch.Tensor):
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / scale_factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
            new_freqs.append((1 - smooth) * freq / scale_factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    return torch.view_as_real(xq_ * freqs_cis).flatten(3).type_as(xq), torch.view_as_real(xk_ * freqs_cis).flatten(3).type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return x[:, :, :, None, :].expand(bs, slen, n_kv_heads, n_rep, head_dim).reshape(bs, slen, n_kv_heads * n_rep, head_dim)


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

        self.cache_k[:bsz, start_pos : start_pos + seqlen] = xk
        self.cache_v[:bsz, start_pos : start_pos + seqlen] = xv

        keys = self.cache_k[:bsz, : start_pos + seqlen]
        values = self.cache_v[:bsz, : start_pos + seqlen]

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
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
    

class KAN(nn.Module):
    def __init__(self, args: ModelArgs, hidden_dim: int):
        super().__init__()
        self.w1 = KANLinear(args.dim, hidden_dim)
        self.w3 = KANLinear(args.dim, hidden_dim)
        self.w2 = KANLinear(hidden_dim, args.dim)

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
            self.mlp = KAN(args, 4 * args.dim)
        else:
            self.mlp = MLP(args, 4 * args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.mlp(self.mlp_norm(h))
        return out



class Llama3_1Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.freqs_cis = precompute_freqs_cis(args.dim // args.n_heads, args.max_seq_len * 2, args.rope_theta, args.use_scaled_rope)

        self.embeddings = nn.Embedding(args.vocab_size, args.dim, padding_idx=args.padding_idx)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(TransformerBlock(layer_id, args))

        self.norm = RMSNorm(args.dim, args.rms_norm_eps)
        self.lm_head = nn.Linear(args.dim, args.vocab_size, bias=False)

    @torch.inference_mode()
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
            loss = None

        return logits, loss


model = Llama3_1Transformer(ModelArgs)
print(model)

input = torch.tensor([[0, 1, 4, 12, 9]])
out = model(input)
print(out)