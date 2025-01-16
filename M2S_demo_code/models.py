import torch
import copy
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from einops import rearrange, repeat

class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super(Encoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src):
        output = src

        for i in range(self.num_layers):
            output = self.layers[i](output)

        if self.norm:
            output = self.norm(output)

        return output

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=1024, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.layers = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.drop_out(src2)
        src = self.norm1(src)

        src2 = self.layers(src)
        src = src + src2
        src = self.norm2(src)

        return src

class EEG_EncoderLayer(nn.Module):
    def __init__(self, d_model=310, nhead=4, dim_feedforward=1024, dropout=0.1):
        super(EEG_EncoderLayer, self).__init__()
        self.self_attn = STSAttention(query_dim=5, heads=nhead, dim_head=64)
        self.layers = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop_out = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src)
        src = src + self.drop_out(src2)
        src = self.norm1(src)

        src2 = self.layers(src)
        src = src + src2
        src = self.norm2(src)

        return src

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class STSAttention(nn.Module):
    def __init__(self, query_dim, heads, dim_head, dropout=0.0, time_window=5):
        super(STSAttention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.time_window = time_window
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.scale = dim_head ** -0.5

        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(inner_dim, query_dim))
        self.to_out.append(nn.Dropout(dropout))

    def reshape_heads_to_batch_dim(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
        return tensor

    def reshape_batch_dim_to_heads(self, tensor):
        batch_size, seq_len, dim = tensor.shape
        head_size = self.heads
        tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
        tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
        return tensor

    def _attention(self, query, key, value, attention_mask=None):
        # print("q:", query.shape)
        # print("k:", key.shape)
        # print("v:", value.shape)

        attention_scores = torch.baddbmm(
            torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
            query,
            key.transpose(-1, -2),
            beta=0,
            alpha=self.scale,
        )
        # print("attention_scores:", attention_scores.shape)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = attention_scores.softmax(dim=-1)
        # cast back to the original dtype
        attention_probs = attention_probs.to(value.dtype)

        # compute attention output
        hidden_states = torch.bmm(attention_probs, value)
        # reshape hidden_states
        hidden_states = self.reshape_batch_dim_to_heads(hidden_states)
        return hidden_states

    def forward(self, hidden_states):
        hidden_states = rearrange(hidden_states, 'b n (c d) -> (b n) c d', c=62)
        query = self.to_q(hidden_states)
        # print(query.shape)
        query = self.reshape_heads_to_batch_dim(query)

        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)
        former_frame_index = torch.arange(self.time_window) - 1
        former_frame_index[0] = 0
        # print("key:", key.shape)

        key = rearrange(key, "(b f) d c -> b f d c", f=self.time_window)
        # print("key:", key.shape)
        key = (key + key[:, [0] * self.time_window] + key[:, former_frame_index]) / 3
        key = rearrange(key, "b f d c -> (b f) d c")

        value = rearrange(value, "(b f) d c -> b f d c", f=self.time_window)
        value = (value + value[:, [0] * self.time_window] + value[:, former_frame_index]) / 3
        value = rearrange(value, "b f d c -> (b f) d c")

        key = self.reshape_heads_to_batch_dim(key)
        value = self.reshape_heads_to_batch_dim(value)

        hidden_states = self._attention(query, key, value)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        hidden_states = rearrange(hidden_states, "(b n) c d -> b n (c d)", n=self.time_window)
        return hidden_states

