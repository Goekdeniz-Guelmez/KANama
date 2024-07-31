from typing import Optional

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.kan import KANLinear
from model.args import ModelArgs
from model.utils import RMSNorm, precompute_freqs_cis, apply_rotary_emb, repeat_kv


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



class KANamav1(nn.Module):
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

# # Define a simple dataset and dataloader
# input_data = torch.tensor([[0, 1, 4, 12, 9]], dtype=torch.long)
# target_data = torch.tensor([[1, 4, 12, 9, 0]], dtype=torch.long)
# dataset = torch.utils.data.TensorDataset(input_data, target_data)
# dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

# # Define the model, loss function, and optimizer
# model = Llama3_1Transformer(ModelArgs())
# print(model)
# out = model(input_data)
# print(out)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# # Training loop
# num_epochs = 100

# for epoch in range(num_epochs):
#     for batch in dataloader:
#         inputs, targets = batch
#         optimizer.zero_grad()
#         outputs, loss = model(inputs, targets=targets)
        
#         loss.backward()
#         optimizer.step()

#         # Update grid points at the end of each epoch
#         for layer in model.layers:
#             if isinstance(layer.mlp, KANLinear):
#                 with torch.no_grad():
#                     for batch in dataloader:
#                         inputs, _ = batch
#                         layer.mlp.update_grid(inputs)
        
#     print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')