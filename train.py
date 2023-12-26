import os
import json
from time import strftime, localtime, time

import torch
from torch import nn
from torch.utils.data import DataLoader

import util.data as Data
from util.args import arg_parser
from model import load_model, head, model_tot, contrastive_model

from util.perf_calc import conf_matrix_calc, metric_calc

from util.loss import loss
import re
import random
import numpy as np


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore


## argparse
args = arg_parser()
seed = random.randint(1, 40000)
if not args.seed and args.seed != -1:
    args.seed = seed
seed_everything(args.seed)
args.contrastive = True  # if args.alpha != 0 else False
## Dataset information
if args.dataset == "ACP2_main":
    tr_file = "./dataset/ACPred-LAF/ACP2_main_train.csv"
    test_file = "./dataset/ACPred-LAF/ACP2_main_test.csv"
elif args.dataset == "ACP2_alter":
    tr_file = "./dataset/ACPred-LAF/ACP2_alternate_train.csv"
    test_file = "./dataset/ACPred-LAF/ACP2_alternate_test.csv"
elif args.dataset == "LEE_Indep":
    tr_file = "./dataset/ACPred-LAF/LEE_Dataset.csv"
    test_file = "./dataset/ACPred-LAF/Independent dataset.csv"
elif args.dataset == "ACP500_ACP164":
    tr_file = "./dataset/ACPred-LAF/ACP_FL_train_500.csv"
    test_file = "./dataset/ACPred-LAF/ACP_FL_test_164.csv"
elif args.dataset == "ACP500_ACP2710":
    tr_file = "./dataset/ACPred-LAF/ACPred-Fuse_ACP_Train500.csv"
    test_file = "./dataset/ACPred-LAF/ACPred-Fuse_ACP_Test2710.csv"
elif args.dataset == "ACP_Mixed_80":
    tr_file = "./dataset/ACPred-LAF/ACP-Mixed-80-train.csv"
    test_file = "./dataset/ACPred-LAF/ACP-Mixed-80-test.csv"
else:
    raise ValueError("Correct dataset type is not provided.")

### File information
tm = localtime(time())
t = strftime("%Y%m%d_%H%M%S", tm)

path_base = os.path.join("./save/", args.dataset)
path_log = os.path.join(path_base, f"best_perf_{args.id}.json")
path_model = os.path.join(path_base, "model")
file_model_temp = os.path.split(args.model_info)[-1].replace(".json", "")
file_model = os.path.join(path_model, f"{file_model_temp}_{t}")
os.makedirs(path_model, exist_ok=True)
### File information end

### Data load
x_tr, y_tr, x_val, y_val = Data.raw_data_read(tr_file, args.val_fold, seed=args.seed)
x_test, y_test = Data.raw_data_read(test_file)
start_token = True if args.model == "encoder" else False

# if args.contrastive:
data_t1 = Data.dataset((x_tr, y_tr), 1, start_token=start_token)
data_t2 = Data.dataset((x_tr, y_tr), 2, start_token=start_token)
data_train = Data.pretrain_dataset(data_t1, data_t2)

data_v1 = Data.dataset((x_val, y_val), 1, start_token=start_token)
data_v2 = Data.dataset((x_val, y_val), 2, start_token=start_token)
data_valid = Data.pretrain_dataset(data_v1, data_v2)

data_te1 = Data.dataset((x_test, y_test), 1, start_token=start_token)
data_te2 = Data.dataset((x_test, y_test), 2, start_token=start_token)
data_test = Data.pretrain_dataset(data_te1, data_te2)

if args.beta == 0:
    args.AA_tok_len = 1
    with open(f"{file_model}_1.json", "wt") as f:
        json.dump(vars(args), f, indent=4)
elif args.beta == 1:
    args.AA_tok_len = 2
    with open(f"{file_model}_2.json", "wt") as f:
        json.dump(vars(args), f, indent=4)
else:
    args.AA_tok_len = 1
    with open(f"{file_model}_1.json", "wt") as f:
        json.dump(vars(args), f, indent=4)

    args.AA_tok_len = 2
    with open(f"{file_model}_2.json", "wt") as f:
        json.dump(vars(args), f, indent=4)

#### If CNN, max_len is required
if args.model == "lstm":
    collate_fn = Data.collate_fn_lstm(args.contrastive)
    head_inp_size = args.n_hidden * 2 if args.bidirectional else args.n_hidden
elif args.model == "encoder":
    collate_fn = Data.collate_fn_encoder(args.contrastive)
    head_inp_size = args.emb_dim
elif args.model == "cnn1d":
    max_len = max(
        max(data_train.return_max_len(), data_valid.return_max_len()),
        data_test.return_max_len(),
    )
    collate_fn = Data.collate_fn_cnn(max_len, args.contrastive)
    head_inp_size = args.channels[-1]
tr_loader = DataLoader(
    data_train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn
)
val_loader = DataLoader(
    data_valid, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
)
if test_file:
    test_loader = DataLoader(
        data_test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

### Device setting
if args.gpu == -1:
    args.device = torch.device("cpu")
else:
    args.device = torch.device(f"cuda:{args.gpu}")
### Device setting end

tr_loader = Data.device_DataLoader(tr_loader, args.device, args.contrastive)
val_loader = Data.device_DataLoader(val_loader, args.device, args.contrastive)
test_loader = Data.device_DataLoader(test_loader, args.device, args.contrastive)
### Data load end

### Model builder

args.AA_tok_len = 1
args.vocab_size = data_t1.vocab_size()
l_feature1 = load_model(args.model)(**args)

args.AA_tok_len = 2
args.vocab_size = data_t2.vocab_size()
l_feature2 = load_model(args.model)(**args)
l_proj1 = head(
    inp_size=head_inp_size,
    output_size=head_inp_size / 4,
    hidden=[head_inp_size / 2],
)
l_proj2 = head(
    inp_size=head_inp_size,
    output_size=head_inp_size / 4,
    hidden=[head_inp_size / 2],
)
l_lin_head1 = head(inp_size=head_inp_size, output_size=2)
l_lin_head2 = head(inp_size=head_inp_size, output_size=2)


model1 = model_tot(feat=l_feature1, linear_head=l_lin_head1, projector=l_proj1)
model2 = model_tot(feat=l_feature2, linear_head=l_lin_head2, projector=l_proj2)
model = contrastive_model(model1, model2).to(args.device)


### Learning parameters define
loss_fn = loss(
    n_cls=2,
    temp=args.temp,
    contrastive=args.contrastive,
    beta=args.beta,
    alpha=args.alpha,
).to(device=args.device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

if args.scheduler:
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda epoch: 0.95**epoch, verbose=False
    )
else:
    scheduler = None
### Learning parameters define end


def train_loop(
    dataloader,
    model,
    loss_fn,
    optimizer,
    scheduler,
    contrastive=False,
    test=False,
    beta=None,
):
    loss_total = 0

    n_batches = len(dataloader)

    if not test:
        model.train()
        for batch in dataloader:
            logit = model(batch)
            loss = loss_fn(logit, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_total += loss
        if scheduler is not None:
            scheduler.step()
        loss = loss_total / n_batches
        return loss
    else:
        model.eval()
        with torch.no_grad():
            if contrastive and (beta == 0.5):
                conf_mat = torch.zeros(2, 2, 2)
            else:
                conf_mat = torch.zeros(1, 2, 2)

            for batch in dataloader:
                logit = model(batch)
                if contrastive and (beta == 0):
                    conf_mat[0] += conf_matrix_calc(logit[0]["lin_head"], batch[0]["y"])
                elif contrastive and (beta == 1):
                    conf_mat[0] += conf_matrix_calc(logit[1]["lin_head"], batch[1]["y"])
                elif contrastive and (beta == 0.5):
                    conf_mat[0, :, :] += conf_matrix_calc(
                        logit[0]["lin_head"], batch[0]["y"]
                    )
                    conf_mat[1, :, :] += conf_matrix_calc(
                        logit[1]["lin_head"], batch[1]["y"]
                    )
                else:
                    conf_mat[0, :, :] += conf_matrix_calc(logit["lin_head"], batch["y"])
            perf_res = []
            for i in range(conf_mat.size(0)):
                perf_res.append(metric_calc(conf_mat[i], return_conf=True))
        return perf_res


val_best_MCC = [0, 0]
best_perf = {}
for ep in range(args.epoch):
    tr_loss = train_loop(
        dataloader=tr_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        contrastive=args.contrastive,
    )
    val_res = train_loop(
        dataloader=val_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        contrastive=args.contrastive,
        test=True,
        beta=args.beta,
    )
    test_res = train_loop(
        dataloader=test_loader,
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        contrastive=args.contrastive,
        test=True,
        beta=args.beta,
    )

    for i, v in enumerate(val_res):
        if val_best_MCC[i] < v["mcc"]:
            val_best_MCC[i] = v["mcc"]
            best_perf_temp = {"epoch": ep}  # , "val_perf": v, "test_perf": test_res[i]}
            val_perf_temp = {f"val_perf_{k}": v for k, v in v.items()}
            test_perf_temp = {f"test_perf_{k}": v for k, v in test_res[i].items()}
            best_perf_temp.update(val_perf_temp)
            best_perf_temp.update(test_perf_temp)
            # with open(f"{file_model}_1.json", "wt") as f:
            if args.contrastive and args.beta == 0:
                torch.save(model1.state_dict(), f"{file_model}_1.pt")
                _id = os.path.split(f"{file_model}_1")[-1]
            elif args.contrastive and args.beta == 1:
                torch.save(model2.state_dict(), f"{file_model}_2.pt")
                _id = os.path.split(f"{file_model}_2")[-1]
            elif args.contrastive and args.beta == 0.5 and i == 0:
                torch.save(model1.state_dict(), f"{file_model}_1.pt")
                _id = os.path.split(f"{file_model}_1")[-1]
            elif args.contrastive and args.beta == 0.5 and i == 1:
                torch.save(model2.state_dict(), f"{file_model}_2.pt")
                _id = os.path.split(f"{file_model}_2")[-1]
            else:
                torch.save(model.state_dict(), f"{file_model}.pt")
                _id = os.path.split(f"{file_model}")[-1]
            best_perf[_id] = best_perf_temp

## Result save
if os.path.isfile(path_log):
    with open(path_log, "r") as f_json:
        bp = json.load(f_json)
else:
    bp = {}
with open(path_log, "w") as f_json:
    for k, v in best_perf.items():
        bp[k] = v
    json.dump(bp, f_json, indent=4)
