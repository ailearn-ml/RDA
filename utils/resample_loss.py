# This code is modified from https://github.com/wutong16/DistributionBalancedLoss
# and https://github.com/blessu/BalancedLossNLP

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np


class ResampleLoss(nn.Module):

    def __init__(self,
                 loss_weight=1.0, reduction='mean',
                 reweight_func=None,  # None, 'inv', 'sqrt_inv', 'rebalance', 'CB'
                 weight_norm=None,  # None, 'by_instance', 'by_batch'
                 focal=dict(
                     focal=True,
                     alpha=0.5,
                     gamma=2, ),
                 map_param=dict(
                     alpha=10.0,
                     beta=0.2,
                     gamma=0.1
                 ),
                 CB_loss=dict(
                     CB_beta=0.9,
                     CB_mode='average_w'  # 'by_class', 'average_n', 'average_w', 'min_n'
                 ),
                 logit_reg=dict(
                     neg_scale=5.0,
                     init_bias=0.1
                 ),
                 class_freq=None,
                 train_num=None):
        super(ResampleLoss, self).__init__()

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.cls_criterion = kl_divergence
        # reweighting function
        self.reweight_func = reweight_func
        # normalization (optional)
        self.weight_norm = weight_norm
        # focal loss params
        self.focal = focal['focal']
        self.gamma = focal['gamma']
        self.alpha = focal['alpha']  # change to alpha
        # mapping function params
        self.map_alpha = map_param['alpha']
        self.map_beta = map_param['beta']
        self.map_gamma = map_param['gamma']
        # CB loss params (optional)
        self.CB_beta = CB_loss['CB_beta']
        self.CB_mode = CB_loss['CB_mode']

        self.class_freq = torch.from_numpy(np.asarray(class_freq)).float()
        self.class_freq[self.class_freq == 0] = 1  # to prevent zero in class_freq
        self.num_classes = self.class_freq.shape[0]
        self.train_num = train_num  # only used to be divided by class_freq
        # regularization params
        self.logit_reg = logit_reg
        self.neg_scale = logit_reg['neg_scale'] if 'neg_scale' in logit_reg else 1.0
        init_bias = logit_reg['init_bias'] if 'init_bias' in logit_reg else 0.0
        self.init_bias = - torch.log(self.train_num / self.class_freq - 1) * init_bias
        self.freq_inv = torch.ones(self.class_freq.shape) / self.class_freq
        self.propotion_inv = self.train_num / self.class_freq

    def forward(self,
                cls_score,
                label,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        if weight is None:
            weight = self.reweight_functions(label)

        cls_score, weight = self.logit_reg_functions(label.float(), cls_score, weight)

        if self.focal:
            pt = torch.exp(cls_score - torch.max(cls_score, dim=1, keepdim=True)[0])
            pt = pt / torch.sum(pt, dim=1, keepdim=True)
            loss = self.cls_criterion(
                cls_score, label.float(), weight=weight, reduction='none')
            loss = ((1 - pt) ** self.gamma) * loss
            loss = reduce_loss(loss, reduction)
        else:
            loss = self.cls_criterion(cls_score, label.float(), weight,
                                      reduction=reduction)

        loss = self.loss_weight * loss
        return loss

    def reweight_functions(self, label):
        if self.reweight_func is None:
            return None
        elif self.reweight_func in ['inv', 'sqrt_inv']:
            weight = self.RW_weight(label.float())
        elif self.reweight_func in 'rebalance':
            weight = self.rebalance_weight(label.float())
        elif self.reweight_func in 'CB':
            weight = self.CB_weight(label.float())
        else:
            return None

        if self.weight_norm is not None:
            if 'by_instance' in self.weight_norm:
                max_by_instance, _ = torch.max(weight, dim=-1, keepdim=True)
                weight = weight / max_by_instance
            elif 'by_batch' in self.weight_norm:
                weight = weight / torch.max(weight)
        return weight

    def logit_reg_functions(self, labels, logits, weight=None):
        if not self.logit_reg:
            return logits, weight
        if 'init_bias' in self.logit_reg:
            logits = logits + self.init_bias.to(logits.device)
        if 'neg_scale' in self.logit_reg:
            logits = logits * (1 - labels) * self.neg_scale + logits * labels
            if weight is not None:
                weight = weight / self.neg_scale * (1 - labels) + weight * labels
        return logits, weight

    def rebalance_weight(self, gt_labels):
        repeat_rate = torch.sum(gt_labels.float() * self.freq_inv.to(gt_labels.device), dim=1, keepdim=True)
        pos_weight = self.freq_inv.clone().detach().unsqueeze(0).to(gt_labels.device) / repeat_rate
        # pos and neg are equally treated
        weight = torch.sigmoid(self.map_beta * (pos_weight - self.map_gamma)) + self.map_alpha
        return weight

    def CB_weight(self, gt_labels):
        if 'by_class' in self.CB_mode:
            weight = torch.tensor((1 - self.CB_beta)) / (1 - torch.pow(self.CB_beta, self.class_freq))
        elif 'average_n' in self.CB_mode:
            avg_n = torch.sum(gt_labels * self.class_freq, dim=1, keepdim=True) / torch.sum(gt_labels, dim=1,
                                                                                            keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)) / (1 - torch.pow(self.CB_beta, avg_n))
        elif 'average_w' in self.CB_mode:
            weight_ = torch.tensor((1 - self.CB_beta)) / (1 - torch.pow(self.CB_beta, self.class_freq))
            weight = torch.sum(gt_labels * weight_, dim=1, keepdim=True) / torch.sum(gt_labels, dim=1, keepdim=True)
        elif 'min_n' in self.CB_mode:
            min_n, _ = torch.min(gt_labels * self.class_freq +
                                 (1 - gt_labels) * 100000, dim=1, keepdim=True)
            weight = torch.tensor((1 - self.CB_beta)) / \
                     (1 - torch.pow(self.CB_beta, min_n))
        else:
            raise NameError
        return weight

    def RW_weight(self, gt_labels, by_class=True):
        if 'sqrt' in self.reweight_func:
            weight = torch.sqrt(self.propotion_inv)
        else:
            weight = self.propotion_inv
        if not by_class:
            sum_ = torch.sum(weight * gt_labels, dim=1, keepdim=True)
            weight = sum_ / torch.sum(gt_labels, dim=1, keepdim=True)
        return weight


def kl_divergence(pred, label, weight=None, reduction='mean', avg_factor=None):
    pred = torch.exp(pred - torch.max(pred, dim=1, keepdim=True)[0])
    pred = pred / torch.sum(pred, dim=1, keepdim=True)
    loss = label * torch.log(torch.maximum(label / torch.maximum(pred, torch.tensor([1e-14]).to(label.device)),
                                           torch.tensor([1e-14]).to(label.device)))

    # apply weights and do the reduction
    if weight is not None:
        weight = weight.to(loss.device).float()
    loss = weight_reduce_loss(loss, weight=weight, reduction=reduction, avg_factor=avg_factor)
    return loss


def reduce_loss(loss, reduction):
    """Reduce loss as specified.
    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".
    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()


def weight_reduce_loss(loss, weight=None, reduction='mean', avg_factor=None):
    """Apply element-wise weight and reduce loss.
    Args:
        loss (Tensor): Element-wise loss.
        weight (Tensor): Element-wise weights.
        reduction (str): Same as built-in losses of PyTorch.
        avg_factor (float): Avarage factor when computing the mean of losses.
    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            loss = loss.sum() / avg_factor
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss
