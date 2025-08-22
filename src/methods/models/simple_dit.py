import torch
from torch import nn

class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True)
        return (x - mean) / (std + self.eps)

class SelfAttention(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5
        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x):
        B, L, D = x.shape
        qkv = self.to_qkv(x).reshape(B, L, 3, self.heads, D // self.heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, L, D)
        return self.to_out(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim):
        super().__init__()
        self.ln1 = LayerNorm(dim)
        self.attn = SelfAttention(dim, heads)
        self.ln2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.ReLU(),
            nn.Linear(mlp_dim, dim),
        )

    def forward(self, x):
        x = self.ln1(x)
        x = x + self.attn(x)
        x = self.ln2(x)
        x = x + self.mlp(x)
        return x

class Embedding(nn.Module):
    def __init__(self, dim, num_categories):
        super().__init__()
        self.embedding = nn.Parameter(torch.empty((num_categories, dim)))
        torch.nn.init.kaiming_uniform_(self.embedding, a=5**0.5)

    def forward_differentiable(self, x):
        return x @ self.embedding
    
    def forward(self, x):
        return self.embedding[x]
    
class CondEmbedding(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.mlp = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.SiLU(),
            nn.Linear(output_size, output_size)
        )

    def forward(self, t: torch.Tensor):
        if len(t.shape) == 1:
            t = t.unsqueeze(-1).repeat(1, self.input_size)
        return self.mlp(t.float())

class DiT(nn.Module):
    def __init__(self, dim, num_categories):
        super().__init__()
        self.dim = dim
        self.num_categories = num_categories

        self.hidden_size = 256
        self.heads = 8
        self.mlp_dim = 512
        self.embedding = Embedding(self.hidden_size, num_categories)
        self.cond_embedding = CondEmbedding(self.hidden_size, dim*self.hidden_size)
        self.net = nn.Sequential(
            TransformerBlock(self.hidden_size, self.heads, self.mlp_dim),
            TransformerBlock(self.hidden_size, self.heads, self.mlp_dim),
            TransformerBlock(self.hidden_size, self.heads, self.mlp_dim),
            nn.Linear(self.hidden_size, self.num_categories),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if len(x.squeeze().shape) > 2:
            x = self.embedding.forward_differentiable(x)
        else:
            x = self.embedding(x)
        out = x + self.cond_embedding(t).reshape(x.shape[0], x.shape[1], self.hidden_size)
        out = self.net(out)
        return out