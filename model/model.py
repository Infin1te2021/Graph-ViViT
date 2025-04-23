import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat

from vit.transformer_module import NodeEmbedding, Transformer, FactorizedTransformer
from vit.bias import GraphAttnBiasSpatial, GraphAttnBiasTemporal, CentralityEncoder

def exists(val):
  return val is not None

def pair(t):
  return t if isinstance(t, tuple) else (t, t)


class VideoViT_GraphEmbd_STB(nn.Module):
  def __init__(self, *, image_size, image_patch_size, frames, frame_patch_size, num_classes, dim, spatial_depth, temporal_depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0., variant = 'factorized_encoder', spatial_bias=True, temporal_bias=True, emb_method='111'):
    super().__init__()
    image_height, image_width = pair(tuple(image_size))
    patch_height, patch_width = pair(tuple(image_patch_size))

    assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'
    assert frames % frame_patch_size == 0, 'Frames must be divisible by frame patch size'
    assert variant in ('factorized_encoder', 'factorized_self_attention'), f'variant = {variant} is not implemented'
    assert dim % (patch_height * patch_width * frame_patch_size) == 0 

    num_image_patches = (image_height // patch_height) * (image_width // patch_width)
    num_frame_patches = (frames // frame_patch_size)

    patch_dim = channels * patch_height * patch_width * frame_patch_size

    assert len(emb_method) == 3, 'emb_method must be a string of 3 characters'
    self.node_enabled = (emb_method[0] == '1')
    self.velocity_enabled = (emb_method[1] == '1')
    self.centr_enabled = (emb_method[2] == '1')
    assert any([self.node_enabled, self.velocity_enabled, self.centr_enabled]), "at least one embedding method must be enabled"
    if self.centr_enabled:
      central_dim = int(dim / (patch_height * patch_width * frame_patch_size))
      self.central_encoder = CentralityEncoder(max_degree=4, dim=central_dim, patch_height=patch_height, patch_width=patch_width, frame_patch_size=frame_patch_size)

    assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

    self.global_average_pool = pool == 'mean'

    if spatial_bias:
      self.spatial_attn_bias = GraphAttnBiasSpatial(num_heads=heads, spatial_bias_hidden_dim=64, multi_hop_max_dist=4, n_layers=3, num=patch_width, frames=frames, frame_patch_size=frame_patch_size)
    else:
      self.spatial_attn_bias = None

    if temporal_bias:
      self.temporal_attn_bias = GraphAttnBiasTemporal(num_heads=heads, temporal_hidden_dim=64, num_frame_patches=num_frame_patches)
    else:
      self.temporal_attn_bias = None

    if self.node_enabled:
      self.to_patch_embedding_nodeFeature = NodeEmbedding(patch_dim, proj_hidden_dim=64, dim=dim, patch_height=patch_height, patch_width=patch_width, frame_patch_size=frame_patch_size)
    
    if self.velocity_enabled:
      self.to_patch_embedding_nodeVelocity = NodeEmbedding(patch_dim, proj_hidden_dim=64, dim=dim, patch_height=patch_height, patch_width=patch_width, frame_patch_size=frame_patch_size)

    self.pos_embedding = nn.Parameter(torch.randn(1, num_frame_patches, num_image_patches, dim))

    self.dropout = nn.Dropout(emb_dropout)

    self.spatial_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None

    if variant == 'factorized_encoder':
      self.temporal_cls_token = nn.Parameter(torch.randn(1, 1, dim)) if not self.global_average_pool else None
      self.spatial_transformer = Transformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout, drop_path=0.2)
      self.temporal_transformer = Transformer(dim, temporal_depth, heads, dim_head, mlp_dim, dropout, drop_path=0.2)
    elif variant == 'factorized_self_attention':
      assert spatial_depth == temporal_depth, 'Spatial and temporal depth must be the same for factorized self-attention'
      self.factorized_transformer = FactorizedTransformer(dim, spatial_depth, heads, dim_head, mlp_dim, dropout)

    self.pool = pool
    self.to_latent = nn.Identity()

    self.mlp_head = nn.Linear(dim, num_classes)
    self.variant = variant

  def forward(self, x):
    
    if self.spatial_attn_bias is not None:
      spatial_attn_bias = self.spatial_attn_bias(x)

    if self.temporal_attn_bias is not None:
      temporal_attn_bias = self.temporal_attn_bias(x)

    if self.velocity_enabled:
      velocity = torch.zeros_like(x)
      velocity[:, :, 1:, :, :] = torch.diff(x, dim=2)
      v = self.to_patch_embedding_nodeVelocity(velocity)

    if self.centr_enabled:
      c = self.central_encoder(x)

    if self.node_enabled:
      x = self.to_patch_embedding_nodeFeature(x)
    
    ## Need to be refactored
    if self.node_enabled:
        if self.velocity_enabled:
            x += v
        if self.centr_enabled:
            x += c
    elif self.velocity_enabled:
        x = v
        if self.centr_enabled:
            x += c
    elif self.centr_enabled:
        x = c
        
    b, f, n, _ = x.shape

    x = x + self.pos_embedding[:, :f, :n]

    if exists(self.spatial_cls_token):
      spatial_cls_tokens = repeat(self.spatial_cls_token, '1 1 d -> b f 1 d', b = b, f = f)
      x = torch.cat((spatial_cls_tokens, x), dim = 2)
    
    x = self.dropout(x)

    if self.variant == 'factorized_encoder':
      x = rearrange(x, 'b f n d -> (b f) n d')
      
      x = self.spatial_transformer(x, attn_bias=spatial_attn_bias)
      
      x = rearrange(x, '(b f) n d -> b f n d', b = b)

      x = x[:, :, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b f d', 'mean')

      if exists(self.temporal_cls_token):
        temporal_cls_tokens = repeat(self.temporal_cls_token, '1 1 d-> b 1 d', b = b)

        x = torch.cat((temporal_cls_tokens, x), dim = 1)
      
      x = self.temporal_transformer(x, attn_bias=temporal_attn_bias)
      
      x = x[:, 0] if not self.global_average_pool else reduce(x, 'b f d -> b d', 'mean')
    
    elif self.variant == 'factorized_self_attention':
      x = self.factorized_transformer(x)
      x = x[:, 0, 0] if not self.global_average_pool else reduce(x, 'b f n d -> b d', 'mean')

    x = self.to_latent(x)

    final_outupt = self.mlp_head(x)

    return final_outupt

  
if __name__ == "__main__":
  model = VideoViT_GraphEmbd_STB(image_size=[25,2], image_patch_size=[1, 1], frames=64, frame_patch_size=4, num_classes=60, dim=96, spatial_depth=6, temporal_depth=6, heads=8, mlp_dim=384, pool='cls', channels=3, dim_head=32, dropout=0.1, emb_dropout=0.1, variant='factorized_encoder', 
                                 spatial_bias=True, 
                                 temporal_bias=True, 
                                 emb_method='010')
  x = torch.randn((16, 3, 64, 25, 2), dtype=torch.float32)  # [Batch, Channel, Frame, Joint, Number of Human]
  output = model(x)
  # torch.onnx.export(model, x, "VideoViT_GraphEmbd_STB.onnx", input_names=["input"], dynamo=True)