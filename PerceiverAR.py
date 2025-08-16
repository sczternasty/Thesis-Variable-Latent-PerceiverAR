

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    def __init__(self, emb_size, heads, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads

        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.size()
        H = self.heads
        S = self.head_dim

        q = self.query(x).view(B, T, H, S).transpose(1, 2).contiguous().view(B * H, T, S)
        k = self.key(x).view(B, T, H, S).transpose(1, 2).contiguous().view(B * H, T, S)
        v = self.value(x).view(B, T, H, S).transpose(1, 2).contiguous().view(B * H, T, S)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(S)

        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
        scores = scores.masked_fill(mask == 1, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v).view(B, H, T, S)
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        return self.out(out)


class CrossAttention(nn.Module):
    def __init__(self, emb_size, heads, dropout=0.1):
        super().__init__()
        self.emb_size = emb_size
        self.heads = heads
        self.head_dim = emb_size // heads

        self.norm = nn.LayerNorm(emb_size)
        self.context_norm = nn.LayerNorm(emb_size)

        self.query = nn.Linear(emb_size, emb_size, bias=False)
        self.key = nn.Linear(emb_size, emb_size, bias=False)
        self.value = nn.Linear(emb_size, emb_size, bias=False)
        self.out = nn.Linear(emb_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, context_mask=None):
        x = self.norm(x)
        context = self.context_norm(context)

        B, T_x, C = x.size()
        _, T_c, _ = context.size()
        H = self.heads
        S = self.head_dim

        q = self.query(x).view(B, T_x, H, S).transpose(1, 2).contiguous().view(B * H, T_x, S)
        k = self.key(context).view(B, T_c, H, S).transpose(1, 2).contiguous().view(B * H, T_c, S)
        v = self.value(context).view(B, T_c, H, S).transpose(1, 2).contiguous().view(B * H, T_c, S)

        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(S)

        if context_mask is not None:
            mask = context_mask.unsqueeze(1).unsqueeze(1).expand(B, 1, T_x, T_c)
            mask = mask.repeat(H, 1, 1, 1).view(B * H, T_x, T_c)
            scores = scores.masked_fill(~mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        out = torch.bmm(attn, v).view(B, H, T_x, S)
        out = out.transpose(1, 2).contiguous().view(B, T_x, S)

        return self.out(out)


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


class PerceiverAR(nn.Module):
    def __init__(self, vocab_size, max_seq_len, emb_size=512, heads=8,
                 num_layers=6, perceive_depth=1, dropout=0.1):
        super().__init__()
        assert emb_size % heads == 0

        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size
        self.emb_size = emb_size
        self.perceive_depth = perceive_depth

        self.token_emb = nn.Embedding(vocab_size, emb_size)
        self.pos_emb = nn.Embedding(max_seq_len, emb_size)
        self.emb_dropout = nn.Dropout(dropout)

        self.cross_attn = CrossAttention(emb_size, heads, dropout)

        hidden_dim = int(emb_size * 4)
        self.perceiver_ff = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, hidden_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, emb_size, bias=False)
        )

        self.transformer_layers = nn.ModuleList([
            TransformerBlock(emb_size, heads, dropout)
            for _ in range(num_layers)
        ])

        self.to_logits = nn.Linear(emb_size, vocab_size, bias=False)

    def forward(self, x, cross_seq_len, mask=None):
        seq_len, device = x.shape[1], x.device
        assert cross_seq_len < seq_len <= self.max_seq_len

        x = self.token_emb(x)
        positions = self.pos_emb(torch.arange(seq_len, device=device)).unsqueeze(0).expand_as(x)
        x = self.emb_dropout(x + positions)

        input_seq = x

        for _ in range(self.perceive_depth):
            n = x[:, -cross_seq_len:]
            cross_attended = self.cross_attn(n, input_seq, mask)
            ff_out = self.perceiver_ff(x[:, -cross_seq_len:] + cross_attended)

            x_new = x.clone()
            x_new[:, -cross_seq_len:] = x[:, -cross_seq_len:] + cross_attended + ff_out
            x = x_new

        for transformer_layer in self.transformer_layers:
            x = transformer_layer(x)

        logits = self.to_logits(x)
        return F.log_softmax(logits, dim=2)