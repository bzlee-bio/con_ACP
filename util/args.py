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

    parser.add_argument("--model_info", type=str, help="Model info data")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--dropout_rate", type=float, default=0.1, help="Dropout rate")
    parser.add_argument("--gpu", type=int, default=-1, help="GPU1 num")
    parser.add_argument("--id", type=int, default=-1, help="GPU1 num")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epoch", type=int, default=500)
    parser.add_argument(
        "--scheduler", type=lambda s: s.lower() in ["true", "1"], default=False
    )
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--val_fold", type=int, default=0)

    # parser.add_argument(
    #     "--load_weight",
    #     type=lambda s: s.lower() in ["true", "1"],
    #     default=False,
    # )
    # parser.add_argument("--pretrained_epoch", type=str)
    ## Pretrain parameter
    # parser.add_argument(
    #     "--n_out_feat", type=int, help="Dim of output features for pretraining"
    # )
    parser.add_argument("--temp", type=float, default=1)
    parser.add_argument(
        "--AA_tok_len",
        type=int,
        default=1,
        help="Length of AA token for prediction classifier",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Loss ratio between contrastive loss 1 and 2",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0,
        help="Loss ratio between contrastive loss and cross-entropy loss ",
    )
    parser.add_argument("--seed", type=int, default=-1, help="random seed num")

    args = AttrDict(vars(parser.parse_args()))

    with open(args.model_info, "rt") as f:
        args_temp = json.load(f)
        args_temp = AttrDict(args_temp)

    args = args_temp + args
    # assert args.tgt_model == None
    # ) and args.contrastive, "Contrastive learning requires tgt_model info"
    # args += AttrDict(args_temp)

    # args.pretrain = False

    ## Binary classification
    # args.n_cls = 1
    # print(args.pretrain_model)
    # if args.pretrain_model:
    #     args_temp = args

    #     with open(
    #         args.pretrain_model.replace("_pretrain", "").replace(".pt", ".json"), "rt"
    #     ) as f:
    #         args = json.load(f)
    #     args = AttrDict(args)
    #     args.batch_size = args_temp.batch_size
    #     args.dataset = args_temp.dataset
    #     args.gpu_num = args_temp.gpu_num
    #     args.lr = args_temp.lr
    #     args.epoch = args_temp.epoch
    #     args.scheduler = args_temp.scheduler
    #     args.val_fold = args_temp.val_fold
    #     args.pretrain_model = args_temp.pretrain_model
    #     args.pretrain = False
    #     args.reset_weight = args_temp.reset_weight
    return args
