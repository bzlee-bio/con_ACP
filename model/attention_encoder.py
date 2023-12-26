import torch
from torch import nn, Tensor


class multihead_attention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(multihead_attention, self).__init__()
        assert d_model % h == 0
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.h = h
        self.d_k = d_model // h

        self.attn_prob = None

        if dropout is not None:
            self.dropout = nn.Dropout(p=dropout)

    def attention(self, q: Tensor, k: Tensor, v: Tensor, mask=None, dropout=None):
        # q, k (..., seq_len, d_k)
        # v (..., seq_len, d_v)
        attn_score = (
            torch.matmul(q, k.transpose(-1, -2)) / k.size(-1) ** 0.5
        )  # QK^T/sqrt(d_k)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e9)
        attn_prob = torch.softmax(attn_score, dim=-1)
        if dropout is not None:
            attn_prob = dropout(attn_prob)
        return torch.matmul(attn_prob, v), attn_prob

    def forward(self, q, k, v, mask=None):
        # q, k, v (..., seq_len, d_model)
        n_batch = q.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1)

        q = self.w_q(q)
        k = self.w_k(k)
        v = self.w_v(v)
        q = q.view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        k = k.view(n_batch, -1, self.h, self.d_k).transpose(1, 2)
        v = v.view(n_batch, -1, self.h, self.d_k).transpose(1, 2)

        x, self.attn_prob = self.attention(q, k, v, mask)
        x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        return self.w_o(x)


class position_wise_feedforward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1) -> None:
        super(position_wise_feedforward, self).__init__()
        self.lin1 = nn.Linear(d_model, d_ff)
        self.lin2 = nn.Linear(d_ff, d_model)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.lin1(x)
        x = self.relu(x)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lin2(x)


class residual_connect(nn.Module):
    def __init__(self, d_model, dropout=0.1) -> None:
        """Residual connection & layer normalization

        Args:
            inp_shape (list or torch.size): _description_
        """
        super(residual_connect, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

    def forward(self, x, sub_layer):
        identity = x
        x = sub_layer(self.layer_norm(x))
        x = self.layer_norm(x)
        if self.dropout is not None:
            x = self.dropout(x)

        return x + identity


class encoder_layer(nn.Module):
    def __init__(self, h, d_model, d_ff, dropout) -> None:
        super(encoder_layer, self).__init__()
        self.self_attn = multihead_attention(h, d_model, dropout)
        self.pwff = position_wise_feedforward(
            d_model=d_model, d_ff=d_ff, dropout=dropout
        )
        self.dropout = dropout

        self.res_conn1 = residual_connect(d_model=d_model, dropout=dropout)
        self.res_conn2 = residual_connect(d_model=d_model, dropout=dropout)

    def forward(self, x, mask):
        x = self.res_conn1(x, lambda x: self.self_attn(x, x, x, mask))
        x = self.res_conn2(x, self.pwff)
        return x


class encoder(nn.Module):
    def __init__(self, num_encoder, h, d_model, d_ff, dropout) -> None:
        super().__init__()
        self.encoders = nn.ModuleList(
            [
                encoder_layer(h=h, d_model=d_model, d_ff=d_ff, dropout=dropout)
                for _ in range(num_encoder)
            ]
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        for layer in self.encoders:
            x = layer(x, mask)
        return self.layer_norm(x)


class embed_tok(nn.Module):
    def __init__(self, vocab_size, d_model, **kwargs):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # self.pos_embed = nn.Embedding(252, d_model, padding_idx=0)

    def forward(self, x, **kwargs):
        # x_pos = self.pos_embed(seq_idx)
        x = self.embed(x)
        return x


class embed_pos(nn.Module):
    def __init__(self, d_model, **kwargs):
        super().__init__()
        self.pos_embed = nn.Embedding(252, d_model, padding_idx=0)

    def forward(self, seq_idx, **kwargs):
        x_pos = self.pos_embed(seq_idx)
        return x_pos


class embed_layer(nn.Module):
    def __init__(self, vocab_size, d_model, **kwargs):
        super().__init__()
        self.embed = embed_tok(vocab_size=vocab_size, d_model=d_model)
        self.pos_embed = embed_pos(d_model=d_model)

    def forward(self, x, seq_idx, **kwargs):
        x_pos = self.pos_embed(seq_idx)
        x = self.embed(x)
        return x + x_pos


class enc_classifier(nn.Module):
    def __init__(self, n_encoder, h, emb_dim, d_ff, dropout_rate, vocab_size, **kwargs):
        super(enc_classifier, self).__init__()
        d_model = emb_dim
        self.embed_layer = embed_layer(vocab_size=vocab_size, d_model=d_model)

        self.enc = encoder(
            num_encoder=n_encoder,
            h=h,
            d_model=d_model,
            d_ff=d_ff,
            dropout=dropout_rate,
        )
        # self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        # self.pos_embed = nn.Embedding(252, d_model, padding_idx=0)
        self.dropout = nn.Dropout(dropout_rate)
        self.idx = torch.tensor(list(range(250)), dtype=torch.int32)
        self.out_dim = d_model

    def forward(self, x, seq_idx, mask=None, **kwargs):
        # x_pos = self.pos_embed(seq_idx)
        # x = self.embed(x)

        # x += x_pos
        x = self.embed_layer(x, seq_idx)
        x = self.enc(x, mask)
        return x[:, 0, :].squeeze()  # classification task


# class enc_classifier(nn.Module):
#     def __init__(self, n_encoder, h, emb_dim, d_ff, dropout_rate, vocab_size, **kwargs):
#         d_model = emb_dim
#         super(enc_classifier, self).__init__()
#         self.enc = encoder(
#             num_encoder=n_encoder,
#             h=h,
#             d_model=d_model,
#             d_ff=d_ff,
#             dropout=dropout_rate,
#         )
#         self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
#         self.pos_embed = nn.Embedding(252, d_model, padding_idx=0)
#         self.dropout = nn.Dropout(dropout_rate)
#         self.idx = torch.tensor(list(range(250)), dtype=torch.int32)
#         self.out_dim = d_model

#     def forward(self, x, seq_idx, mask=None, **kwargs):
#         x_pos = self.pos_embed(seq_idx)
#         x = self.embed(x)
#         x += x_pos
#         x = self.enc(x, mask)
#         return x[:, 0, :].squeeze()  # classification task
