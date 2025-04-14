import torch
from torch import nn
from timm.layers import DropPath
from einops import rearrange
from einops.layers.torch import Rearrange

class NodeEmbedding(nn.Module):
  def __init__(self, patch_dim, proj_hidden_dim, dim, patch_height, patch_width, frame_patch_size):
    super().__init__()

    self.net = nn.Sequential(
      Rearrange('b c (f pf) (h p1) (w p2) -> b f (w h) (p1 p2 pf c)', p1 = patch_height, p2 = patch_width, pf = frame_patch_size),
      nn.LayerNorm(patch_dim),
      nn.Linear(patch_dim, proj_hidden_dim),
      nn.ReLU(),
      nn.Linear(proj_hidden_dim, dim),
      nn.LayerNorm(dim)
    )

  def forward(self, x):
    return self.net(x)

class FeedForward(nn.Module):
  def __init__(self, dim, hidden_dim, dropout = 0.):
    super().__init__()
    self.net = nn.Sequential(
      nn.LayerNorm(dim),
      nn.Linear(dim, hidden_dim),
      nn.GELU(),
      nn.Dropout(dropout),
      nn.Linear(hidden_dim, dim),
      nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)

class Attention(nn.Module):
  def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
    super().__init__()
    inner_dim = dim_head *  heads
    project_out = not (heads == 1 and dim_head == dim)

    self.heads = heads
    self.scale = dim_head ** -0.5

    self.norm = nn.LayerNorm(dim)

    self.attend = nn.Softmax(dim = -1)
    self.dropout = nn.Dropout(0.5)

    self.to_q = nn.Linear(dim, inner_dim, bias=False)
    self.to_k = nn.Linear(dim, inner_dim, bias=False)
    self.to_v = nn.Linear(dim, inner_dim, bias=False)

    self.to_out = nn.Sequential(
      nn.Linear(inner_dim, dim),
      nn.Dropout(dropout)
    ) if project_out else nn.Identity()

  def forward(self, x, attn_bias=None):
    x = self.norm(x)

    q = self.to_q(x)
    k = self.to_k(x)
    v = self.to_v(x)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v))

    dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

    if attn_bias is not None:
      dots += attn_bias

    attn = self.attend(dots)
    attn = self.dropout(attn)

    out = torch.matmul(attn, v)
    out = rearrange(out, 'b h n d -> b n (h d)')
    return self.to_out(out)

class Transformer(nn.Module):
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., drop_path=0.):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(nn.ModuleList([
        Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
        FeedForward(dim, mlp_dim, dropout = dropout)
      ]))

  def forward(self, x, attn_bias=None):
    for attn, ff in self.layers:
      x = self.drop_path(attn(x,  attn_bias=attn_bias)) + x
      x = self.drop_path(ff(x)) + x

    return self.norm(x)
    
class FactorizedTransformer(nn.Module):
  def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
    super().__init__()
    self.norm = nn.LayerNorm(dim)
    self.layers = nn.ModuleList([])
    for _ in range(depth):
      self.layers.append(nn.ModuleList([
        Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
        Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
        FeedForward(dim, mlp_dim, dropout = dropout)
      ]))

  def forward(self, x):
    b, f, n, _ = x.shape
    for spatial_attn, temporal_attn, ff in self.layers:
      x = rearrange(x, 'b f n d -> (b f) n d')
      x = spatial_attn(x) + x
      x = rearrange(x, '(b f) n d -> (b n) f d', b=b, f=f)
      x = temporal_attn(x) + x
      x = ff(x) + x
      x = rearrange(x, '(b n) f d -> b f n d', b=b, n=n)

    return self.norm(x)