# Thanks to:
# https://github.com/karpathy/nanoGPT

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """
    def __init__(self, ndim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)
        # output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(input_dim, input_dim))
                                        .view(1, 1, input_dim, input_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class MLP(nn.Module):
    def __init__(
        self,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.c_fc    = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(
        self,
        input_dim: int = 1024,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__()
        self.ln_1 = LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(
            input_dim, n_head, n_embd, dropout, bias
        )
        self.ln_2 = LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_categories: int, 
        num_timesteps: int,
        n_layer: int = 12,
        n_head: int = 12,
        n_embd: int = 768,
        dropout: float = 0.0,
        bias: bool = True
    ) -> None:
        super().__init__()
        assert input_dim is not None
        assert num_categories is not None
        # since we have fixed sequence lenghts and small dims,
        # we can define model with full context size (input and output)
        self.input_dim = 2 * input_dim 
        self.num_categories = num_categories
        self.num_timesteps = num_timesteps

        self.transformer = nn.ModuleDict(dict(
            token_embed = nn.Embedding(num_categories, n_embd),
            pos_embed = nn.Embedding(self.input_dim, n_embd),
            time_embed = nn.Embedding(num_timesteps + 2, n_embd),
            drop = nn.Dropout(dropout),
            h = nn.ModuleList([Block(self.input_dim, n_head, n_embd, dropout, bias) for _ in range(n_layer)]),
            ln_f = LayerNorm(n_embd, bias=bias),
        ))
        self.lm_head = nn.Linear(n_embd, num_categories, bias=False)
        # https://paperswithcode.com/method/weight-tying
        self.transformer.token_embed.weight = self.lm_head.weight 
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        device = x.device
        _, dim = x.shape
        assert dim <= self.input_dim, f"Cannot forward sequence of length {dim}, block size is only {self.input_dim}"
        pos = torch.arange(0, dim, dtype=torch.long, device=device)
        tok_emb = self.transformer.token_embed(x)
        pos_emb = self.transformer.pos_embed(pos)
        time_emb = self.transformer.time_embed(t.unsqueeze(dim=-1).expand_as(x))
        # print(f'tok_ebm.shape={tok_emb.shape}, pos_emb.shape={pos_emb.shape}, time_emb.shape={time_emb.shape}')
        x = self.transformer.drop(tok_emb + pos_emb + time_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x[:, [-1], :])

        return logits