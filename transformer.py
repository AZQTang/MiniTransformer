
# add all  your Encoder and Decoder code here
import torch.nn as nn
import math
import torch
import torch.nn.functional as F
from alibi import get_alibi_bias
class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, alibi=False):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd must be divisible by n_head")
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.alibi = alibi

        # query, key, value projections
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        # projection layer
        self.proj = nn.Linear(n_embd, n_embd, bias=False)

        # alibi bias
        self.register_buffer("alibi_bias", None, persistent=False)

    def forward(self, x):
        B, T, C = x.shape # 16, 32, 64

        # query, key, value (16, 32, 64) -> (16, 32, 8, 8) -> (16, 8, 32, 8)
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # attention scores
        # q - (16, 8, 32, 8)
        # k.transpose(-2,-1) - (16, 8, 8, 32)
        # (q @ k.transpose(-2,-1)) - (16, 8, 32, 32)
        attention_scores = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))

        # add alibi bias
        if self.alibi: 
            if self.alibi_bias is None or self.alibi_bias.shape[-1] != T:
                self.alibi_bias = get_alibi_bias(
                    self.n_head, T, x.device, x.dtype, is_decoder=False
                )  # [1, n_head, T, T]
            attention_scores = attention_scores + self.alibi_bias
        attention_scores = F.softmax(attention_scores, dim=-1)
        # v - (16, 8, 32, 8)
        # attention_scores - (16, 8, 32, 32)
        # attention_scores @ v - (16, 8, 32, 8)
        # .transpose(1, 2) - (16, 32, 8, 8)
        # .contiguous() - (16, 32, 8, 8)
        # .view(B, T, C) - (16, 32, 64)
        y = (attention_scores @ v).transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y, attention_scores
        
class FeedForward(nn.Module):
    """A feedforward network used in the transformer block after the attention layer"""
    def __init__(self, n_embd, dropout=0.2):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.ffn(x)
        


class TransformerBlock(nn.Module):
    """A single transformer block"""
    def __init__(self, n_embd, n_head, alibi=False):
        super().__init__()
        self.attention = MultiHeadAttention(n_embd, n_head, alibi)
        self.feed_forward = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        
    def forward(self, x):
        # Layer Normalization + Multi-Head Attention
        attn, attn_weights = self.attention(self.ln1(x))
        # Residual Connection
        x = x + attn
        # Layer Normalization + Feed Forward
        x = x + self.feed_forward(self.ln2(x))
        return x, attn_weights

class Encoder(nn.Module):
    def __init__(self, vocab_size, n_embd, block_size, n_layer, n_head, alibi=False):
        super().__init__()
        self.alibi = alibi
        # token embedding
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        
        # position embedding
        if not self.alibi:
            self.position_embedding = nn.Embedding(block_size, n_embd)
        else: # add zero position embedding
            #self.position_embedding = nn.Parameter(torch.zeros(1, block_size, n_embd))
            self.position_embedding = None
        # transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(n_embd, n_head, alibi) for _ in range(n_layer)
        ])
        self.ln_final = nn.LayerNorm(n_embd)
        # initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx):
        B, T = idx.shape # (16, 32)
        token_embeddings = self.token_embedding(idx) # (16, 32, 64)
        if not self.alibi:
            position = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # (1, 32)
            position_embeddings = self.position_embedding(position) # (1, 32, 64)
            x = token_embeddings + position_embeddings # (16, 32, 64)
        else:
            x = token_embeddings

        
        # Transformer Blocks
        all_attention_weights = []
        for block in self.blocks:
            x, attn_weights = block(x)
            all_attention_weights.append(attn_weights)
        x = self.ln_final(x)
        
        return x, all_attention_weights

class Classifier(nn.Module):
    # ffn for classification task
    def __init__(self, n_input, n_hidden, n_output):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output),
        )
    def forward(self, x):
        return self.ffn(x)


class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, alibi=False):
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd must be divisible by n_head")
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        self.alibi = alibi

        # query, key, value projections
        self.query = nn.Linear(n_embd, n_embd, bias=False)
        self.key = nn.Linear(n_embd, n_embd, bias=False)
        self.value = nn.Linear(n_embd, n_embd, bias=False)
        # projection layer
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.register_buffer("alibi_bias", None, persistent=False)
        self.register_buffer("mask", torch.tril(torch.ones(1,1,100,100)))
    def forward(self, x):
        B,T,C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        attention_scores = (q @ k.transpose(-2,-1)) * (1.0 / math.sqrt(self.head_dim))
        if self.alibi:
            if self.alibi_bias is None or self.alibi_bias.shape[-1] != T:
                self.alibi_bias = get_alibi_bias(
                    self.n_head, T, x.device, x.dtype, is_decoder=True
                ) 
            attention_scores = attention_scores + self.alibi_bias
        else: 
            mask = self.mask[:,:,:T,:T]
            attention_scores = attention_scores.masked_fill(mask == 0, float('-inf'))
        attention_scores = F.softmax(attention_scores, dim=-1)
        y = attention_scores @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)
        return y, attention_scores

class DecoderFeedForward(nn.Module):
    def __init__(self, n_embd, n_hidden=100, dropout=0.3):
        super().__init__()
        
        self.ffn = nn.Sequential(
            nn.Linear(n_embd, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.ffn(x)

class DecoderBlock(nn.Module):
    def __init__(self, n_embd, n_head, n_hidden, dropout=0.3, alibi=False):
        super().__init__()
        self.attention = MaskedMultiHeadAttention(n_embd, n_head, alibi)
        self.feed_forward = DecoderFeedForward(n_embd, n_hidden, dropout)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
    def forward(self, x):
        attention, attention_weights = self.attention(self.ln1(x))
        x = x + attention
        x = x + self.feed_forward(self.ln2(x))
        return x, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, alibi=False):
        super().__init__()
        self.alibi = alibi
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        if not self.alibi:
            self.position_embedding = nn.Embedding(block_size, n_embd)
        else:
            self.position_embedding = None
        self.blocks = nn.ModuleList([
            DecoderBlock(n_embd, n_head, n_hidden=100, dropout=0.2, alibi=alibi) for _ in range(n_layer)

        ])
        self.ln_final = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

        self.apply(self._init_weights)
        self.block_size = block_size
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError("Cannot forward sequence longer than block size")
        token_embeddings = self.token_embedding(idx)
        if not self.alibi:
            position = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0) # (1, 32)
            position_embeddings = self.position_embedding(position) # (1, 32, 64)
            x = token_embeddings + position_embeddings # (16, 32, 64)
        else:
            x = token_embeddings
        
        all_attention_weights = []
        for block in self.blocks:
            x, attention_weights = block(x)
            all_attention_weights.append(attention_weights)
        x = self.ln_final(x)
        logits = self.lm_head(x) # [B, T, vocab_size]
        if targets is not None:
            B, T, V = logits.shape
            logits = logits.view(B * T, V)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return loss
        else:
            return logits, all_attention_weights
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:,-self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    



    
