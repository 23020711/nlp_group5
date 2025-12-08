import torch
import torch.nn as nn

from conponents import PositionalEncoding, MultiHeadAttention, PositionwiseFeedForward


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        # Sublayer 1: Multi-Head Self-Attention
        _src = self.self_attn(src, src, src, src_mask)
        # Add & Norm
        src = self.norm1(src + self.dropout(_src))

        # Sublayer 2: Feed Forward
        _src = self.ffn(src)
        # Add & Norm
        src = self.norm2(src + self.dropout(_src))

        return src


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head)
        self.cross_attn = MultiHeadAttention(d_model, n_head)
        self.ffn = PositionwiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # 1. Masked Multi-Head Self-Attention
        # trg_mask ở đây đảm bảo vị trí t không nhìn thấy t+1
        _trg = self.self_attn(trg, trg, trg, trg_mask)
        trg = self.norm1(trg + self.dropout(_trg))

        # 2. Multi-Head Cross-Attention
        # Query lấy từ Decoder (trg), Key & Value lấy từ Encoder (enc_src)
        _trg = self.cross_attn(trg, enc_src, enc_src, src_mask)
        trg = self.norm2(trg + self.dropout(_trg))

        # 3. Feed Forward
        _trg = self.ffn(trg)
        trg = self.norm3(trg + self.dropout(_trg))

        return trg

class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, n_layer, n_head, d_ff, dropout, max_len):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        src = self.dropout(self.pos_encoding(self.embedding(src)))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, n_layer, n_head, d_ff, dropout, max_len):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)])
        self.fc_out = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.dropout(self.pos_encoding(self.embedding(trg)))
        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output


class Transformer(nn.Module):
    def __init__(self, src_vocab_size, trg_vocab_size,
                 d_model=512, n_head=8, n_layer=6, d_ff=2048,
                 dropout=0.1, max_len=100):
        super(Transformer, self).__init__()

        self.encoder = Encoder(src_vocab_size, d_model, n_layer, n_head, d_ff, dropout, max_len)
        self.decoder = Decoder(trg_vocab_size, d_model, n_layer, n_head, d_ff, dropout, max_len)

    def make_src_mask(self, src, src_pad_idx):
        # Mask các vị trí padding trong source
        # src: [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
        return (src != src_pad_idx).unsqueeze(1).unsqueeze(2)

    def make_trg_mask(self, trg, trg_pad_idx):
        # Mask padding trong target
        trg_pad_mask = (trg != trg_pad_idx).unsqueeze(1).unsqueeze(2)

        # Mask look-ahead (tam giác trên)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()

        return trg_pad_mask & trg_sub_mask

    def forward(self, src, trg, src_pad_idx=0, trg_pad_idx=0):
        src_mask = self.make_src_mask(src, src_pad_idx)
        trg_mask = self.make_trg_mask(trg, trg_pad_idx)

        enc_src = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_src, trg_mask, src_mask)

        return output