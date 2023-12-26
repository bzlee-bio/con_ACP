from torch import nn
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTM(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim,
        n_hidden,
        n_RNN_layers,
        device,
        dropout_rate=0.5,
        bidirectional=False,
        **kwargs
    ):
        super(LSTM, self).__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=n_hidden,
            num_layers=n_RNN_layers,
            batch_first=True,
            dropout=dropout_rate,
            bidirectional=bidirectional,
        )
        self.out_dim = n_hidden * 2 if bidirectional else n_hidden
        # self.fc = nn.Linear(self.out_size, n_cls)
        self.device = device

    def forward(self, x, seq_len, **kwargs):
        emb = self.embed(x)

        packed_x = pack_padded_sequence(
            emb, seq_len.cpu(), batch_first=True, enforce_sorted=False
        )
        rnn_out, _ = self.rnn(packed_x)
        rnn_out, s_len = pad_packed_sequence(rnn_out, batch_first=True)
        s_idx = (s_len.to(self.device) - 1).unsqueeze(1).unsqueeze(2)
        s_idx = s_idx.expand(s_idx.size(0), 1, rnn_out.size(2))
        gather_res = torch.gather(rnn_out, 1, s_idx)
        gather_res = torch.squeeze(gather_res)
        # return self.fc(gather_res)
        return gather_res
