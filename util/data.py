from string import ascii_uppercase
from typing import Any
import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import Dataset
import numpy as np
import random
import os
from Bio import SeqIO


class dataset(Dataset):
    """
    Args:
        dataset (_type_): Dataset preprocess & loader
    """

    def __init__(self, data, AA_tok_len=1, start_token=False):
        """
        Args:
            data (_type_):
            AA_tok_len (int, optional): Defaults to 1.
            start_token (bool, optional): True is required for encoder-based model.
        """
        ## Load train dataset
        x_data, self.label = data
        self.tok_AA_len = AA_tok_len
        self.default_AA = list("RHKDESTNQCGPAVILMFYW")
        # AAs which are not included in default_AA
        self.wo_AA = [
            AA
            for AA in list(ascii_uppercase)
            if AA not in self.default_AA and AA != "X"
        ]
        self.tokens = self._token_gen(self.tok_AA_len)
        self.tokens += ["<START>"]
        self.token_to_idx = {k: i + 1 for i, k in enumerate(self.tokens)}
        self.token_to_idx["<PAD>"] = 0  ## idx as 0 is PAD
        self.X = [self._AA_to_idx(seq, start_token=start_token) for seq in x_data]
        self.max_len = 0
        for seq in self.X:
            self.max_len = len(seq) if self.max_len < len(seq) else self.max_len

    def _token_gen(self, tok_AA_len: int, st: str = "", curr_depth: int = 0):
        """Generate tokens based on default amino acid residues
            and also includes "X" as arbitrary residues.
            Length of AAs in each token should be provided by "tok_AA_len"

        Args:
            tok_AA_len (int): Length of token
            st (str, optional): Defaults to ''.
            curr_depth (int, optional): Defaults to 0.

        Returns:
            List: List of tokens
        """
        curr_depth += 1
        if curr_depth <= tok_AA_len:
            l = [
                st + t
                for s in self.default_AA + ["X"]
                for t in self._token_gen(tok_AA_len, s, curr_depth)
            ]
            return l
        else:
            return [st]

    def _AA_to_idx(self, seq: str, start_token):
        """Convert each token to index

        Args:
            seq (str): AA sequence

        Returns:
            list: A list of indexes
        """

        seq_idx = []
        if start_token:
            seq_idx += [self.token_to_idx["<START>"]]

        for i in range(len(seq) - self.tok_AA_len + 1):
            curr_token = seq[i : i + self.tok_AA_len]
            if curr_token not in self.token_to_idx.keys():
                for AA in self.wo_AA:
                    curr_token = curr_token.replace(AA, "X")
            seq_idx.append(self.token_to_idx[curr_token])
        return seq_idx

    def vocab_size(self):
        return len(self.token_to_idx)

    def return_max_len(self):
        return self.max_len

    def __len__(self):
        return len(self.label)

    def __repr__(self):
        return f"Total num of data: {len(self.label)}, # of positive data: {sum(self.label)}, # of negative data: {len(self.label)-sum(self.label)}"

    def __getitem__(self, idx):
        x_batch = self.X[idx]
        label_batch = self.label[idx]
        return x_batch, label_batch


class pretrain_dataset(Dataset):
    def __init__(self, d1, d2) -> None:
        self.d1 = d1
        self.d2 = d2

    def __len__(self):
        return len(self.d1)

    def return_max_len(self):
        return self.d1.return_max_len(), self.d2.return_max_len()

    def __getitem__(self, idx):
        _x, _l = self.d1[idx]
        _x2, _l2 = self.d2[idx]
        return (_x, _x2), (_l, _l2)


class collate_fn_lstm:
    def __init__(self, contrastive=False) -> None:
        self.contrastive = contrastive
        self.inference = inference

    def __call__(self, batch):
        if self.contrastive:
            x_batch1, l_batch1, seq_len1 = [], [], []
            x_batch2, l_batch2, seq_len2 = [], [], []
            for x, label in batch:
                x_batch1.append(torch.tensor(x[0], dtype=torch.int64))
                l_batch1.append(label[0])
                seq_len1.append(len(x[0]))

                x_batch2.append(torch.tensor(x[1], dtype=torch.int64))
                l_batch2.append(label[1])
                seq_len2.append(len(x[1]))
            x_batch1 = pad_sequence(x_batch1, batch_first=True, padding_value=0)
            l_batch1 = torch.tensor(l_batch1, dtype=torch.float32)
            seq_len1 = torch.tensor(seq_len1, dtype=torch.int64)

            x_batch2 = pad_sequence(x_batch2, batch_first=True, padding_value=0)
            l_batch2 = torch.tensor(l_batch2, dtype=torch.float32)
            seq_len2 = torch.tensor(seq_len2, dtype=torch.int64)
            return (
                {"x": x_batch1, "y": l_batch1, "seq_len": seq_len1},
                {"x": x_batch2, "y": l_batch2, "seq_len": seq_len2},
            )
        else:
            x_batch, l_batch, seq_len = [], [], []
            for x, label in batch:
                x_batch.append(torch.tensor(x, dtype=torch.int64))
                l_batch.append(label)
                seq_len.append(len(x))
            x_batch = pad_sequence(x_batch, batch_first=True, padding_value=0)
            l_batch = torch.tensor(l_batch, dtype=torch.float32)
            seq_len = torch.tensor(seq_len, dtype=torch.int64)
            return {"x": x_batch, "y": l_batch, "seq_len": seq_len}


class collate_fn_cnn:
    def __init__(self, max_len, contrastive=False) -> None:
        self.max_len = max_len
        self.contrastive = contrastive

    def __call__(self, batch):
        if self.contrastive:
            l_batch1 = []
            x_batch1 = torch.zeros(len(batch), self.max_len[0], dtype=torch.int64)
            l_batch2 = []
            x_batch2 = torch.zeros(len(batch), self.max_len[1], dtype=torch.int64)
            for i, (x, label) in enumerate(batch):
                print(label[0])
                x_batch1[i, : len(x[0])] = torch.tensor(x[0], dtype=torch.int64)
                l_batch1.append(label[0])
                x_batch2[i, : len(x[1])] = torch.tensor(x[1], dtype=torch.int64)
                l_batch2.append(label[1])

            l_batch1 = torch.tensor(l_batch1, dtype=torch.float32)
            l_batch2 = torch.tensor(l_batch2, dtype=torch.float32)
            return ({"x": x_batch1, "y": l_batch1}, {"x": x_batch2, "y": l_batch2})
        else:
            l_batch = []
            x_batch = torch.zeros(len(batch), self.max_len, dtype=torch.int64)
            for i, (x, label) in enumerate(batch):
                x_batch[i, : len(x)] = torch.tensor(x, dtype=torch.int64)
                l_batch.append(label)

            l_batch = torch.tensor(l_batch, dtype=torch.float32)

            return {"x": x_batch, "y": l_batch}


# def contrastive_label(lab):
#     tot_lab = {}
#     for i, labels in enumerate(lab):
#         for sub_l in labels:
#             if sub_l not in tot_lab:
#                 tot_lab[sub_l] = [i]
#             else:
#                 tot_lab[sub_l].append(i)

#     idx = []
#     for _, val_list in tot_lab.items():
#         for v1 in val_list:
#             for v2 in val_list:
#                 idx.append([v1, v2])

#     return (
#         torch.sparse_coo_tensor(
#             list(zip(*idx)), [True] * len(idx), size=(len(lab), len(lab))
#         )
#         .to_dense()
#         .type(torch.float)
#     )


class collate_fn_encoder:
    def __init__(self, contrastive: bool = False) -> None:
        self.contrastive = contrastive

    def __call__(self, batch):
        if self.contrastive:
            x_batch1, l_batch1, seq_idx1, mask1 = [], [], [], []
            x_batch2, l_batch2, seq_idx2, mask2 = [], [], [], []
            for x, label in batch:
                x_batch1.append(torch.tensor(x[0], dtype=torch.int64))
                l_batch1.append(label[0])
                seq_idx1.append(
                    torch.tensor(range(1, len(x[0]) + 1), dtype=torch.int64)
                )

                mask1.append(torch.tensor([1] * len(x[0]), dtype=torch.int64))

                x_batch2.append(torch.tensor(x[1], dtype=torch.int64))
                l_batch2.append(label[1])
                seq_idx2.append(
                    torch.tensor(range(1, len(x[1]) + 1), dtype=torch.int64)
                )

                mask2.append(torch.tensor([1] * len(x[1]), dtype=torch.int64))
            seq_idx1 = pad_sequence(seq_idx1, batch_first=True, padding_value=0)
            x_batch1 = pad_sequence(x_batch1, batch_first=True, padding_value=0)

            l_batch1 = torch.tensor(l_batch1, dtype=torch.float32)
            mask1 = pad_sequence(mask1, batch_first=True, padding_value=0)
            mask1 = mask1.unsqueeze(-2)  # (n_batch, 1, seq_len)
            seq_idx2 = pad_sequence(seq_idx2, batch_first=True, padding_value=0)
            x_batch2 = pad_sequence(x_batch2, batch_first=True, padding_value=0)

            l_batch2 = torch.tensor(l_batch2, dtype=torch.float32)
            mask2 = pad_sequence(mask2, batch_first=True, padding_value=0)
            mask2 = mask2.unsqueeze(-2)  # (n_batch, 1, seq_len)

            return (
                {"x": x_batch1, "y": l_batch1, "seq_idx": seq_idx1, "mask": mask1},
                {"x": x_batch2, "y": l_batch2, "seq_idx": seq_idx2, "mask": mask2},
            )
        else:
            x_batch, l_batch, seq_idx, mask = [], [], [], []
            for x, label in batch:
                x_batch.append(torch.tensor(x, dtype=torch.int64))
                l_batch.append(label)
                seq_idx.append(torch.tensor(range(1, len(x) + 1), dtype=torch.int64))

                mask.append(torch.tensor([1] * len(x), dtype=torch.int64))

            seq_idx = pad_sequence(seq_idx, batch_first=True, padding_value=0)
            x_batch = pad_sequence(x_batch, batch_first=True, padding_value=0)

            l_batch = torch.tensor(l_batch, dtype=torch.float32)
            mask = pad_sequence(mask, batch_first=True, padding_value=0)
            mask = mask.unsqueeze(-2)  # (n_batch, 1, seq_len)
            return {"x": x_batch, "y": l_batch, "seq_idx": seq_idx, "mask": mask}


# class collate_fn_encoder:
#     def __init__(self, pretrain: bool = False) -> None:
#         self.pretrain = pretrain

#     def __call__(self, batch):
#         x_batch, l_batch, seq_idx, mask = [], [], [], []
#         for x, label in batch:
#             x_batch.append(torch.tensor(x, dtype=torch.int64))
#             l_batch.append(label)
#             seq_idx.append(torch.tensor(range(1, len(x) + 1), dtype=torch.int64))

#             mask.append(torch.tensor([1] * len(x), dtype=torch.int64))

#         seq_idx = pad_sequence(seq_idx, batch_first=True, padding_value=0)
#         x_batch = pad_sequence(x_batch, batch_first=True, padding_value=0)
#         if self.pretrain:
#             l_batch = pretrain_label(l_batch)
#         else:
#             l_batch = torch.tensor(l_batch, dtype=torch.float32)
#         mask = pad_sequence(mask, batch_first=True, padding_value=0)
#         mask = mask.unsqueeze(-2)  # (n_batch, 1, seq_len)
#         return {"x": x_batch, "y": l_batch, "seq_idx": seq_idx, "mask": mask}


# def raw_data_read(file, val_fold=None, pretrain=False):
#     with open(file, "r") as f:
#         x, label = [], []
#         if val_fold is not None:
#             x_val, label_val = [], []

#         for l in f.readlines():
#             if "Fold" in l:
#                 pass
#             else:
#                 l_list = l.strip().split(",")
#                 if int(l_list[2]) == val_fold:
#                     x_val.append(l_list[0])
#                     if pretrain:
#                         # label_val.append([int(x) for x in l_list[1].strip().split("|")])
#                         label_val.append([0])
#                     else:
#                         label_val.append(int(l_list[1]))
#                 else:
#                     x.append(l_list[0])
#                     if pretrain:
#                         # label.append([int(x) for x in l_list[1].strip().split("|")])
#                         label.append([0])
#                     else:
#                         label.append(int(l_list[1]))
#     if val_fold is None:
#         return x, label
#     else:
#         return x, label, x_val, label_val


def raw_data_read(file, val_fold=None, pretrain=False, seed=None, inference=False):
    if seed:
        random.seed(seed)
    with open(file, "r") as f:
        data = []
        if val_fold is not None:
            x_val, label_val = [], []

        for l in f.readlines():
            if "Fold" in l:
                pass
            else:
                l_list = l.strip().split(",")
                data.append(l_list[:2])
                # if int(l_list[2]) == val_fold:
                #     x_val.append(l_list[0])
                #     if pretrain:
                #         # label_val.append([int(x) for x in l_list[1].strip().split("|")])
                #         label_val.append([0])
                #     else:
                #         label_val.append(int(l_list[1]))
                # else:
                #     x.append(l_list[0])
                #     if pretrain:
                #         # label.append([int(x) for x in l_list[1].strip().split("|")])
                #         label.append([0])
                #     else:
                #         label.append(int(l_list[1]))
    if inference:
        x, label = [], []
        print(data)
        for _x, _lab in data:
            x.append(_x)
            label.append(_lab)
        return x, label
    if val_fold is None:
        x, label = [], []
        for _x, _lab in data:
            x.append(_x)
            label.append(int(_lab))
        return x, label
    else:
        x, label, x_val, label_val = [], [], [], []
        random.shuffle(data)
        tr_len = len(data) * 0.9
        for i, (_x, _lab) in enumerate(data):
            if i <= tr_len:
                x.append(_x)
                label.append(int(_lab))
            else:
                x_val.append(_x)
                label_val.append(int(_lab))
        return x, label, x_val, label_val


def raw_data_read_fasta(file):
    x, label, id = [], [], []
    for l in SeqIO.parse(file, "fasta"):
        x.append(str(l.seq))
        label.append(0)
        id.append(l.description)
    return x, label, id


class device_DataLoader:
    def __init__(self, dataloader, device, contrastive=False) -> None:
        self.dataloader = dataloader
        self.device = device
        self.contrastive = contrastive

    def __len__(self):
        return len(self.dataloader)

    def __iter__(self):
        if self.contrastive:
            for batch in self.dataloader:
                yield [
                    {k: v.to(self.device) for k, v in batch[0].items()},
                    {k: v.to(self.device) for k, v in batch[1].items()},
                ]

        else:
            for batch in self.dataloader:
                yield {k: v.to(self.device) for k, v in batch.items()}
            # yield tuple(tensor.to(self.device) for tensor in batch)
