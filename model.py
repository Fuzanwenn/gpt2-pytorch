import torch
import torch.nn as nn 
import torch.nn.functional as F

class LayerNorm(nn.Module): 
  def __init__(self, embed_dim): 
    super().__init__()
    self.scale = nn.Parameter(torch.ones(embed_dim)) 
    self.shift = nn.Parameter(torch.zeros(embed_dim))

  def forward(self, input):
      return F.layer_norm(input, self.scale.shape, self.scale, self.shift, 1e-5)

class SelfAttention(nn.Module): 

  def __init__(self, embed_dim: int, n_head: int, block_size: int): 
    super().__init__()
    self.map_qkv = nn.Linear(embed_dim, 3 * embed_dim)   # old self.c_attn

    self.n_head = n_head
    self.embed_dim = embed_dim

    # register mask so it follows the module's device (cpu/cuda/mps)
    mask = torch.tril(torch.ones(block_size, block_size))
    self.register_buffer("mask", mask.view(1, 1, block_size, block_size))

  def forward(self, x): 
      # x: (B, T, C)
      B, T, C = x.size()
      H = self.n_head
      head_dim = C // H

      # 1. Linear map to get q, k, v: (B, T, 3C) -> three (B, T, C)
      qkv = self.map_qkv(x)
      q, k, v = qkv.split(self.embed_dim, dim=2)

      # 2. Reshape for multi-head:
      #    (B, T, C) -> (B, T, H, head_dim) -> (B, H, T, head_dim)
      q = q.view(B, T, H, head_dim).permute(0, 2, 1, 3)
      k = k.view(B, T, H, head_dim).permute(0, 2, 1, 3)
      v = v.view(B, T, H, head_dim).permute(0, 2, 1, 3)

      # 3. Scaled dot-product attention scores: (B, H, T, T)
      att = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)

      # 4. Causal mask: keep only tokens up to position t
      #    self.mask: (1, 1, block_size, block_size)
      mask = self.mask[:, :, :T, :T].to(x.device)
      att = att.masked_fill(mask == 0, float('-inf'))

      # 5. Softmax over keys dimension
      att = F.softmax(att, dim=-1)

      # 6. Apply attention weights to values: (B, H, T, head_dim)
      out = att @ v

      # 7. Merge heads back: (B, H, T, head_dim) -> (B, T, C)
      out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)

      y = out
      assert y.shape == (B, T, C)
      return y

class MLP(nn.Module): 

  def __init__(self, embed_dim, latent_dim_multiplier): 
    super().__init__()
    self.c_fc    = nn.Linear(embed_dim, latent_dim_multiplier * embed_dim, bias=True)
    self.gelu    = nn.ReLU()
    self.c_proj  = nn.Linear(latent_dim_multiplier * embed_dim, embed_dim, bias=True)

  def forward(self, x): 
    x = self.c_fc(x)
    x = self.gelu(x)
    x = self.c_proj(x)
    return x

class Block(nn.Module): 
  def __init__(self, embed_dim: int, n_head: int, block_size: int): 
    super().__init__()
    self.ln_1 = LayerNorm(embed_dim)
    self.attn = SelfAttention(embed_dim, n_head=n_head, block_size=block_size)
    self.ln_2 = LayerNorm(embed_dim)
    self.mlp = MLP(embed_dim, latent_dim_multiplier=4)

  def forward(self, x):
    x = x + self.attn(self.ln_1(x))
    x = x + self.mlp(self.ln_2(x))
    return x

class Transformer(nn.Module): 

  def __init__(self, 
               vocab_size: int, 
               block_size: int, 
               embed_dim: int, 
               n_layer: int
               ): 
    super().__init__()
    # encoding the input 
    self.token_encoder = nn.Embedding(vocab_size, embed_dim)
    self.position_encoder = nn.Embedding(block_size, embed_dim)
    self.transformer = nn.ModuleList([Block(embed_dim, 4, 1024) for _ in range(n_layer)])
    self.final_layernorm = LayerNorm(embed_dim) 
    self.final_linearmap = nn.Linear(embed_dim, vocab_size)
    self.block_size = block_size

  def forward(self, x: torch.Tensor): 
    # x: (B, T)
    B, T = x.size()

    x = x.to(self.token_encoder.weight.device)  # ensure indices on same device as embeddings

    token_embedding = self.token_encoder(x)  # (B, T, C)
    positions = torch.arange(T, device=x.device)
    position_embedding = self.position_encoder(positions)  # (T, C)

    x = token_embedding + position_embedding  # broadcast over batch

    for block in self.transformer: 
      x = block(x) 

    x = self.final_layernorm(x)
    logits = self.final_linearmap(x)
    return logits

  def sample(self, x, max_tokens):
    for _ in range(max_tokens): 
      # clip the context to the block size
      idx_cond = x if x.size(1) <= self.block_size else x[:, -self.block_size:]
      logits = self(idx_cond)
      logits = logits[:, -1, :] # pluck the logits at the final step 
      probs = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probs, num_samples=1) # sampling for generation
      x = torch.cat((x, idx_next), dim=1)
    return x

