# %%
from torch import nn
import torch

# def cnn1d(in_ch, out_ch, stride=1, padding='same'):


class _cnn1d_block(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        dropout_rate: float,
        down_scale: bool = False,
        kernel_size: int = 3,
    ) -> None:
        super(_cnn1d_block, self).__init__()

        stride = 2 if down_scale else 1

        self.cnn1 = nn.Conv1d(
            in_ch, out_ch, kernel_size, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.cnn2 = nn.Conv1d(
            out_ch, out_ch, kernel_size, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout_rate)

        self.relu = nn.ReLU(inplace=True)
        self.downscale = None
        if down_scale:
            self.downscale = nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        if self.downscale is not None:
            identity = self.downscale(identity)
        out = self.cnn1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.cnn2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out


class cnn_1d(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        num_layer: int,
        channels: list,
        ini_channel: int = 32,
        dropout_rate: float = 0.3,
        **kwargs
    ) -> None:
        super(cnn_1d, self).__init__()

        # assert len(num_layers) == len(channels), "Dim of layers and channels must be equal"
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)

        self.curr_channel = ini_channel
        self.conv1 = nn.Conv1d(
            emb_dim,
            self.curr_channel,
            kernel_size=5,
            stride=1,
            padding="same",
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(self.curr_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.fc = nn.Linear(channels[-1], n_cls)

        self.stacked_layers = []
        for channel in channels:
            self.stacked_layers += self._block_stack(num_layer, channel, dropout_rate)
        self.stacked_layers = nn.Sequential(*self.stacked_layers)
        self.out_dim = channels[-1]

    def _block_stack(self, num_layer: int, channel: int, dropout_rate: float):
        layers = []
        layers.append(_cnn1d_block(self.curr_channel, channel, dropout_rate, True))
        for _ in range(1, num_layer):
            layers.append(_cnn1d_block(channel, channel, dropout_rate))
        self.curr_channel = channel
        return layers

    def forward(self, x, **kwargs):
        x = self.embed(x)
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.stacked_layers(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        return x
