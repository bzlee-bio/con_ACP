import argparse
import json


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

    def __add__(self, y):
        for k, v in y.items():
            self.__dict__[k] = v
        return self


def arg_parser():
    parser = argparse.ArgumentParser("ACP model training..")
    # print(parser)
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for prediction"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="ACP_Mixed_80",
        help="Optimized model selection",
    )
    parser.add_argument("--input", type=str, help="Input fasta file for inference")

    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Usage deivce information Options: cpu or gpu",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./output.csv",
        help="Output file for save prediction results",
    )
    args = AttrDict(vars(parser.parse_args()))
    return args
