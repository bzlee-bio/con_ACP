import torch
import torch.nn.functional as F
from torch import nn


class loss(nn.Module):
    def __init__(
        self,
        n_cls,
        mean=True,
        temp=1,
        contrastive=False,
        beta=0,
        alpha=0.6,
        device="cpu",
    ) -> None:
        """_summary_

        Args:
            n_cls (_type_): _description_
            mean (bool, optional): _description_. Defaults to True.
            temp (int, optional): _description_. Defaults to 1.
            contrastive (bool, optional): _description_. Defaults to False.
            tgt_model (_type_, optional): target model info. None, "first", "second", "both". Defaults to None.
        """
        super().__init__()
        self.temp = temp
        self.mean = mean
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.n_cls = n_cls
        self.contrastive = contrastive
        self.beta = beta
        self.alpha = alpha
        self.device = device

    def ce_loss(self, inp, tgt):
        tgt = F.one_hot(tgt.to(torch.int64), self.n_cls)
        return torch.mean(torch.sum(-tgt * self.logsoftmax(inp), dim=1))

    # def contrastive_loss(self, proj1, proj2, labels=None):
    #     proj1 = F.normalize(proj1, dim=1)
    #     proj2 = F.normalize(proj2, dim=1)
    #     features = torch.cat([proj1.unsqueeze(1), proj2.unsqueeze(1)], dim=1)
    #     batch_size = proj1.shape[0]

    #     if labels is None:
    #         mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
    #     else:
    #         labels = labels.contiguous().view(-1, 1)
    #         mask = torch.eq(labels, labels.T).float().to(self.device)

    #     contrast_count = features.shape[1]
    #     contrast_feature = torch.cat(
    #         torch.unbind(features, dim=1), dim=0
    #     )  # torch.cat([proj1, proj2], dim=0)
    #     print(contrast_feature.shape, features.shape)
    #     anchor_dot_contrast = torch.div(
    #         torch.matmul(contrast_feature, contrast_feature.T), self.temp
    #     )
    #     dot_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
    #     logits = anchor_dot_contrast - dot_max.detach()
    #     # print(mask, mask.shape)
    #     mask = mask.repeat(contrast_count, contrast_count)
    #     logits_mask = torch.ones_like(mask) - torch.eye(mask.shape[0]).to(self.device)
    #     mask *= logits_mask

    #     exp_logits = torch.exp(logits) * logits_mask
    #     log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

    #     mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
    #     print(mean_log_prob_pos)
    #     # print(mask, mask.shape)
    #     # print(anchor_dot_contrast, anchor_dot_contrast.shape)
    #     # dot = torch.matmul(proj1, proj2.T) / self.temp
    #     # dot_max, _ = torch.max(dot, dim=1, keepdim=True)
    #     # dot = dot - dot_max.detach()

    #     # exp_dot = torch.exp(dot)
    #     # log_prob = torch.diag(dot, 0) - torch.log(exp_dot.sum(1))
    #     # cont_loss = -log_prob.mean()
    #     # return cont_loss

    #     return None

    def contrastive_loss(self, proj1, proj2, label=None):
        proj1 = F.normalize(proj1, dim=1)
        proj2 = F.normalize(proj2, dim=1)
        dot = torch.matmul(proj1, proj2.T) / self.temp
        dot_max, _ = torch.max(dot, dim=1, keepdim=True)
        dot = dot - dot_max.detach()

        exp_dot = torch.exp(dot)
        log_prob = torch.diag(dot, 0) - torch.log(exp_dot.sum(1))
        cont_loss = -log_prob.mean()
        return cont_loss

    def forward(self, model_out, batch):
        # if self.contrastive:
        proj1 = model_out[0]["proj"]
        proj2 = model_out[1]["proj"]
        # print(proj1)
        # if self.alpha!=0:
        cont_loss = (
            self.contrastive_loss(proj1, proj2, batch[0]["y"])
            if not self.alpha == 0
            else 0
        )
        # else:
        # cont_loss = None

        # if self.tgt_model == "first":
        #     logit1 = model_out[0]["lin_head"]
        #     ce_loss1 = self.ce_loss(logit1, batch[0]["y"])
        #     return cont_loss * self.alpha + ce_loss1 * (1 - self.alpha)
        # elif self.tgt_model == "second":
        #     logit2 = model_out[1]["lin_head"]
        #     ce_loss2 = self.ce_loss(logit2, batch[1]["y"])
        #     return cont_loss * self.alpha + ce_loss2 * (1 - self.alpha)
        # elif self.tgt_model == "both":
        logit1 = model_out[0]["lin_head"]
        logit2 = model_out[1]["lin_head"]
        ce_loss1 = self.ce_loss(logit1, batch[0]["y"])
        ce_loss2 = self.ce_loss(logit2, batch[1]["y"])

        # print(
        #     (ce_loss1 * (1 - self.beta) + ce_loss2 * self.beta) * (1 - self.alpha)
        #     + self.alpha * (cont_loss)
        # )
        return (ce_loss1 * (1 - self.beta) + ce_loss2 * self.beta) * (
            1 - self.alpha
        ) + self.alpha * (cont_loss)
        # tot_loss2 = cont_loss * self.alpha + ce_loss2 * (1 - self.alpha)
        # return cont_loss * self.alpha + (
        #     (1 - self.beta) * ce_loss1 + self.beta * ce_loss2
        # ) / 2 * (1 - self.alpha)
        # else:
        #     return self.ce_loss(model_out["lin_head"], batch["y"])
