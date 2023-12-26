from .lstm import LSTM
from .cnn import cnn_1d
from .attention_encoder import enc_classifier, encoder
from torch import nn

_models = {"lstm": LSTM, "cnn1d": cnn_1d, "encoder": enc_classifier}


def load_model(key):
    if key in _models:
        return _models[key]
    else:
        raise ValueError(f'Invalid model architecture name, "{key}"')


class head(nn.Module):
    def __init__(self, inp_size, output_size=None, hidden=None, **kwargs):
        super(head, self).__init__()
        if hidden:
            nodes = [inp_size] + hidden + [output_size]
        else:
            nodes = [inp_size] + [output_size]
        self.linear = []
        # print(nodes)
        for i in range(len(nodes) - 1):
            self.linear.append(nn.ReLU())
            self.linear.append(nn.Linear(int(nodes[i]), int(nodes[i + 1])))

        self.linear = nn.ModuleList(self.linear)

    def forward(self, x):
        for layer in self.linear:
            x = layer(x)
        return x


class model_tot(nn.Module):
    def __init__(
        self,
        feat,
        linear_head=None,
        projector=None,
        # feat2=None,
        # linear_head2=None,
        # projector2=None,
    ):
        super(model_tot, self).__init__()
        self.feat = feat
        self.linear_head = linear_head
        self.projector = projector
        # self.feat2 = feat2
        # self.linear_head2 = linear_head2
        # self.projector2 = projector2

    def forward(self, batch):
        x = self.feat(**batch)
        output = {}
        if self.linear_head:
            lh = self.linear_head(x)
            output["lin_head"] = lh
        if self.projector:
            proj = self.projector(x)
            output["proj"] = proj

            # return_data["proj"] = proj
        # if self.feat2:
        #     x2 = self.feat2(**batch2)
        #     lh2 = self.linear_head2(x2)
        #     return_data["lh2"] = lh2
        #     if self.projector2:
        #         proj2 = self.projector2(x2)
        #         return_data["proj2"] = proj2
        return output


class contrastive_model(nn.Module):
    def __init__(self, model1, model2) -> None:
        super().__init__()
        self.model1 = model1
        self.model2 = model2

    def forward(self, batch):
        out1 = self.model1(batch[0])
        # print(out1)
        out2 = self.model2(batch[1])
        # print(out2)
        return out1, out2
