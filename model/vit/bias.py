import torch
import torch.nn as nn
import math
from einops import rearrange, reduce
from torch_geometric.utils import degree

def init_params(module, n_layers):
  if isinstance(module, nn.Linear):
    module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
    if module.bias is not None:
      module.bias.data.zero_()
  if isinstance(module, nn.Embedding):
    module.weight.data.normal_(mean=0.0, std=0.02)

def decrease_to_max_value(x, max_value):
  x[x > max_value] = max_value
  return x

def decrease_to_max_value(x, max_value):
  x[x > max_value] = max_value
  return x

class CentralityEncoder(nn.Module):
  def __init__(self, max_degree, dim, patch_height, patch_width,frame_patch_size):
    super().__init__()

    assert max_degree >= 1, "max_degree must be at least 1"
    self.max_degree = max_degree
    self.dim = dim
    self.patch_height = patch_height
    self.patch_width = patch_width
    self.frame_patch_size = frame_patch_size
    self.z_degree = nn.Parameter(torch.randn((max_degree, dim)))

    edges = [
      [4,5],[5,6],[6,7],[7,21],[7,22], [8,9],[9,10],[10,11],
      [11,23],[11,24],[12,13],[13,14],[14,15], [16,17],[17,18],
      [18,19],[0,1],[1,20],[20,2],[2,3],[0,12],[0,16],[20,4],[20,8]
    ]

    undir_edges = []
    for u, v in edges:
      undir_edges.extend([[u, v], [v, u]])

    self.n_node = 25

    self.edge_index = torch.tensor(undir_edges, dtype=torch.long).t().contiguous()

  def forward(self, x):
    
    batch_size, _, num_frames, num_joints, num_coords = x.shape

    assert num_joints == self.n_node, f"Expected {self.n_node} joints, but got {num_joints}."

    deg = decrease_to_max_value(degree(self.edge_index[0], num_nodes=self.n_node, dtype=torch.long).long(), self.max_degree)

    central_embed = self.z_degree[deg-1]

    expanded_embeddings = central_embed.permute(1, 0)             # [dim, 25]
    expanded_embeddings = expanded_embeddings.unsqueeze(0)        # [1, dim, 25]
    expanded_embeddings = expanded_embeddings.unsqueeze(2)        # [1, dim, 1, 25]
    expanded_embeddings = expanded_embeddings.unsqueeze(-1)       # [1, dim, 1, 25, 1]
    
    expanded_embeddings = expanded_embeddings.expand(batch_size, -1,  num_frames, -1, num_coords)

    expanded_embeddings = rearrange(expanded_embeddings, 'b c (f pf) (h p1) (w p2) -> b f (w h) (p1 p2 pf c)', p1 = self.patch_height, p2 = self.patch_width, pf = self.frame_patch_size)
    return expanded_embeddings
  
class SpatialPosEncoder(nn.Module):
  """Encode 3D Euclidean distances with MLP"""
  def __init__(self, num_heads, spatial_bias_hidden_dim=64):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(1, spatial_bias_hidden_dim),
      nn.ReLU(),
      nn.Linear(spatial_bias_hidden_dim, num_heads)
    )
  
  def forward(self, dist):
    dist = dist.float()
    # dist: [n_graph, n_node, n_node]
    return self.mlp(dist.unsqueeze(-1)).permute(0, 3, 1, 2)  # [n_graph, num_heads, n_node, n_node]

class GraphAttnBiasSpatial(nn.Module):
  def __init__(
    self,
    num_heads=8,
    spatial_bias_hidden_dim=64,
    multi_hop_max_dist=4,  # Max allowed hop distance
    n_layers=3,
    num = 1,
    frames = 64,
    frame_patch_size=1
  ):
    super().__init__()
    self.num_heads = num_heads
    self.multi_hop_max_dist = multi_hop_max_dist
    self.num = num
    self.frame_patch_size = frame_patch_size

    assert frames % frame_patch_size == 0, "Frames must be divisible by frame patch size"
    assert num > 0 and num <= 2, "num must be 1 or 2"

    # Pre-defined edges (fixed skeleton structure)
    edges = [
      [4,5],[5,6],[6,7],[7,21],[7,22], [8,9],[9,10],[10,11],
      [11,23],[11,24],[12,13],[13,14],[14,15], [16,17],[17,18],
      [18,19],[0,1],[1,20],[20,2],[2,3],[0,12],[0,16],[20,4],[20,8]
    ]

    if num == 1:
      self.edges = edges + [[u+25, v+25] for u, v in edges]
      self.n_node = 50
    else:
      self.edges = edges
      self.n_node = 25
    
    # Precompute shortest path distances
    self.register_buffer("dist_matrix", self._compute_shortest_path(), persistent=False)
    
    # Spatial encoder (3D distance)
    self.spatial_pos_encoder = SpatialPosEncoder(num_heads, spatial_bias_hidden_dim)
    
    # Edge encoder (multi-hop)
    self.edge_encoder = nn.Embedding(
      multi_hop_max_dist + 2,  # 0~max_hop + padding(0)
      num_heads,
      padding_idx=0
    )
    
    # self.graph_token_virtual_distance = nn.Parameter(torch.randn(1, num_heads))
    ## (Apr 13 2025) Key modification
    self.graph_token_virtual_distance = nn.Parameter(
        torch.randn(num_heads, self.n_node)  # 形状 [num_heads, 25/50]
    )
    # Initialize parameters
    self.apply(lambda module: init_params(module, n_layers=n_layers))

  def _compute_shortest_path(self):
    """Floyd-Warshall algorithm for 25-node graph"""
    n_node = self.n_node
    INF = 999
    dist = torch.full((n_node, n_node), INF, dtype=torch.long)
    
    # Initialize adjacency matrix
    for i in range(n_node):
      dist[i][i] = 0
    for u, v in self.edges:
      dist[u, v] = 1
      dist[v, u] = 1  # Undirected graph
    
    # Floyd-Warshall
    for k in range(n_node):
      for i in range(n_node):
        for j in range(n_node):
          dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
    
    # Handle unreachable pairs
    dist[dist >= INF] = -1
    return dist  # [25/50, 25/50]

  def forward(self, x):
    B, C, F, H, W = x.shape

    if self.num == 1:
      x_body1 = x[..., 0]
      x_body2 = x[..., 1]
      x = torch.cat((x_body1, x_body2), dim=3)
      x = x.unsqueeze(-1)
      # x = x.view(B, C, F, 50, 1)
    else:
      x = x.sum(dim=4)
      x = x.unsqueeze(4)

    x = rearrange(x, 'b c f h w -> (b f w) h c')  # [B*F*W, 25/50, 3] [Num of Graph, 25/50 joints, 3D coordinates]
    n_graph = x.shape[0]

    # 1. Compute 3D spatial distances
    spatial_dist = torch.cdist(x, x, compute_mode="donot_use_mm_for_euclid_dist")  # [n_graph, 25, 25]

    ### Key modification (Apr 13 2025)
    mean = spatial_dist.mean(dim=(1, 2), keepdim=True)
    std = spatial_dist.std(dim=(1, 2), keepdim=True)
    spatial_dist = (spatial_dist - mean) / (std + 1e-5)
    ###

    spatial_bias = self.spatial_pos_encoder(spatial_dist)  # [n_graph, num_heads, 25/50, 25/50]
    
    # 2. Compute multi-hop bias
    hop_dist = self.dist_matrix.unsqueeze(0).repeat(n_graph, 1, 1)  # [n_graph, 25/50, 25/50]
    
    # Clip hop distances
    if self.multi_hop_max_dist > 0:
      hop_dist = torch.clamp(hop_dist, -1, self.multi_hop_max_dist)
      hop_dist[hop_dist == -1] = 0  # Mask invalid to padding_idx=0
    
    hop_bias = self.edge_encoder(hop_dist).permute(0, 3, 1, 2)  # [n_graph, num_heads, 25/50, 25/50]
    
    num_joints = self.n_node + 1 # 25 joints + 1 virtual node (cls)
    
    attn_bias = torch.zeros(n_graph, self.num_heads, num_joints, num_joints, device=x.device)

    # Encode hop distances
    attn_bias[:, :, 1:, 1:] = spatial_bias + hop_bias
    
    # (Apr 13 2025) Key modification
    # virtual_bias = self.graph_token_virtual_distance.view(1, self.num_heads, 1)
    virtual_bias = self.graph_token_virtual_distance.unsqueeze(0)
    attn_bias[:, :, 1:, 0] += virtual_bias
    attn_bias[:, :, 0, 1:] += virtual_bias
    
    attn_bias = rearrange(attn_bias, '(b f w) h n1 n2 -> b h w f n1 n2', b=B, f=F)  # [n_graph, num_heads, 25, 25] -> [Batch, num_heads, 1, Frame, 26/51, 26/51]
    attn_bias = attn_bias.squeeze(2)  # Remove the frame dimension (1) [Batch, num_heads, Frame, 26/51, 26/51]
    
    attn_bias = rearrange(attn_bias, 'b h (f pf) n1 n2 -> b h f pf n1 n2', pf = self.frame_patch_size).sum(dim=3) # [Batch, num_heads, Frame, 26/51, 26/51] -> [Batch, num_heads, Number of Frame Patch, 26/51, 26/51]
    
    attn_bias = rearrange(attn_bias, 'b h f n1 n2 -> (b f) h n1 n2')  
    return attn_bias


class TemporalPosEncoder(nn.Module):
  """Encode temporal distances with MLP"""
  def __init__(self, num_heads, temporal_bias_hidden_dim=64):
    super().__init__()
    self.mlp = nn.Sequential(
      nn.Linear(1, temporal_bias_hidden_dim),
      nn.ReLU(),
      nn.Linear(temporal_bias_hidden_dim, num_heads)
    )
  
  def forward(self, pos_diff):
    # pos_diff: [n_frames, n_frames] or [batch, n_frames, n_frames]
    return self.mlp(pos_diff.unsqueeze(-1)).permute(0, 3, 1, 2)
    

class GraphAttnBiasTemporal(nn.Module):
  def __init__ (self,num_heads=8,temporal_hidden_dim=64,num_frame_patches=16):
    super().__init__()
    self.num_heads = num_heads

    self.patch_num = num_frame_patches

    self.max_time_span = math.ceil(self.patch_num / 4)

    # Temporal encoder (temporal distance)
    self.temporal_pos_encoder = TemporalPosEncoder(num_heads, temporal_hidden_dim)

    self.register_buffer("pos_matrix", self._create_position_matrix(self.patch_num), persistent=False)

    # (Apr 13 2025) Key modification
    # self.temporal_token_virtual_pos = nn.Parameter(torch.randn(1, num_heads, 1))

    self.temporal_token_virtual_pos = nn.Parameter(
      torch.randn(num_heads, num_frame_patches)  # [H, num_patches]
    )
  
  def _create_position_matrix(self, num_patches):

    pos = torch.arange(num_patches, dtype=torch.float)
    return pos[:, None] - pos[None, :]
  
  def forward(self, x):

    B, C, F, H, W = x.shape

    pos_diff = self.pos_matrix[:self.patch_num, :self.patch_num].abs().unsqueeze(0)
    pos_diff = torch.clamp(pos_diff, 0, self.max_time_span)

    temporal_bias = self.temporal_pos_encoder(pos_diff)

    final_bias = torch.zeros(B, self.num_heads, self.patch_num+1, self.patch_num+1, device=x.device, dtype=x.dtype)

    final_bias[:, :, 1:, 1:] = temporal_bias

    # (Apr 13 2025) Key modification
    # virtual_bias = self.temporal_token_virtual_pos.expand(B, -1, -1)
    virtual_bias = self.temporal_token_virtual_pos.unsqueeze(0)
    final_bias[:, :, 1:, 0] += virtual_bias
    final_bias[:, :, 0, 1:] += virtual_bias

    return final_bias