import torch
import torch.nn as nn
import tqdm
import math
import torch.nn.functional as F

class SelfAttention(nn.Module):
    def __init__(self, emb_size=256, heads=4, dropout=0.1):
        super().__init__()
        self.emb_size, self.heads = emb_size, heads
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.unify_heads = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        H = self.heads

        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        S = C // H

        k = k.view(B, T, H, S)
        q = q.view(B, T, H, S)
        v = v.view(B, T, H, S)

        k = k.transpose(1, 2).contiguous().view(B * H, T, S)
        q = q.transpose(1, 2).contiguous().view(B * H, T, S)
        v = v.transpose(1, 2).contiguous().view(B * H, T, S)

        dot = torch.bmm(q, k.transpose(1, 2))  # B * H, T, S) @ (B * H, S, T) -> (B * H, T, T)

        dot = dot / math.sqrt(S)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        dot = dot.masked_fill(mask == 1, float('-inf'))

        dot = F.softmax(dot, dim=2)

        dot = self.dropout(dot)
        out = torch.bmm(dot, v).view(B, H, T, S)

        out = out.transpose(1, 2).contiguous().view(B, T, S * H)  # (B, H, T, S) -> (B, T, S * H) -> (B, T, C)

        out = self.unify_heads(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, emb_size, heads, dropout=0.1):
        super().__init__()

        self.attention = SelfAttention(emb_size, heads)
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)

        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4 * emb_size),
            nn.ReLU(),
            nn.Linear(4 * emb_size, emb_size),
            nn.Dropout(dropout))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, vocab_size, seq_length, emb_size=256, heads=4, num_layers=6, dropout=0.1):
        super().__init__()
        assert emb_size % heads == 0
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.heads = heads

        self.emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Embedding(seq_length, emb_size)

        self.emb_dropout = nn.Dropout(dropout)
        self.pos_dropout = nn.Dropout(dropout)
        self.final_dropout = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(emb_size, heads, dropout=dropout) for _ in range(num_layers)]
        )

        self.fc = nn.Linear(emb_size, vocab_size)

    def forward(self, x):
        x = self.emb(x)
        x = self.emb_dropout(x)

        B, T, C = x.size()
        positions = self.pos_emb(torch.arange(T, device=x.device)).unsqueeze(0).expand(B, T, C)
        positions = self.pos_dropout(positions)

        x = x + positions

        for transformer in self.transformer_blocks:
            x = transformer(x)

        x = self.final_dropout(x)
        x = x.view(B * T, C)
        x = self.fc(x).view(B, T, self.vocab_size)
        return F.log_softmax(x, dim=2)


