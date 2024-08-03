import torch
from torch import nn
from methods.template import LDLTemplate, kaiming_normal_init_net


class RDA(LDLTemplate):
    def __init__(self, num_feature, num_classes, loss_func, hidden_dim=100,
                 lambda1=0.1, lambda2=0.1, lambda3=0.1,
                 lr=1e-3, weight_decay=1e-4, adjust_lr=False, gradient_clip_value=5.0,
                 max_epoch=None, verbose=False, device='cuda:0'):
        super(RDA, self).__init__(num_feature,
                                  num_classes,
                                  adjust_lr=adjust_lr,
                                  gradient_clip_value=gradient_clip_value,
                                  max_epoch=max_epoch,
                                  verbose=verbose,
                                  device=device)
        self.loss_func = loss_func
        self.hidden_dim = hidden_dim
        self.lambda1, self.lambda2, self.lambda3 = lambda1, lambda2, lambda3

        self.encoder_x = nn.Linear(num_feature, hidden_dim)
        self.encoder_x_mu = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_x_log_var = nn.Linear(hidden_dim, hidden_dim)

        self.encoder_y = nn.Linear(num_classes, hidden_dim)
        self.encoder_y_mu = nn.Linear(hidden_dim, hidden_dim)
        self.encoder_y_log_var = nn.Linear(hidden_dim, hidden_dim)

        self.decoder = nn.Linear(hidden_dim, num_classes)

        kaiming_normal_init_net(self)
        params_decay = (p for name, p in self.named_parameters() if 'bias' not in name)
        params_no_decay = (p for name, p in self.named_parameters() if 'bias' in name)
        self.optimizer = torch.optim.AdamW(
            [{'params': params_decay, 'lr': lr, 'weight_decay': weight_decay},
             {'params': params_no_decay, 'lr': lr}], amsgrad=True)
        self.to(self.device)

    def set_forward(self, x):
        x = self.encoder_x(x)
        x = self.decoder(torch.relu(x))
        x = nn.Softmax(dim=1)(x)
        return x

    def set_forward_loss(self, x, y):
        xx = self.encoder_x(x)
        xx_mu = self.encoder_x_mu(xx)
        xx_log_var = self.encoder_x_log_var(xx)

        rand = torch.normal(mean=0., std=1., size=xx_mu.shape).to(self.device)
        dx = xx_mu + rand * torch.exp(xx_log_var) ** 0.5

        yy = self.encoder_y(y.float())
        yy_mu = self.encoder_y_mu(yy)
        yy_log_var = self.encoder_y_log_var(yy)

        rand = torch.normal(mean=0., std=1., size=yy_mu.shape).to(self.device)
        dy = yy_mu + rand * torch.exp(yy_log_var) ** 0.5

        sigma2 = yy_log_var.detach()
        loss_target = -0.5 * torch.mean(torch.sum(
            xx_log_var - sigma2 - torch.exp(xx_log_var) / torch.exp(sigma2) - (xx_mu - yy_mu.detach()) ** 2 / torch.exp(
                sigma2) + 1, dim=1))
        loss_recovery = nn.CrossEntropyLoss()(self.decoder(dy), y)

        sx = self.cosine_similarity(dx, dx).reshape(-1)
        sy = self.cosine_similarity(dy, dy).reshape(-1)
        loss_similarity = torch.sum(nn.MSELoss()(sx, sy))

        pred = self.decoder(torch.relu(xx))
        loss_pred = self.loss_func(pred, y)

        loss = loss_pred + self.lambda1 * loss_recovery + self.lambda2 * loss_target + self.lambda3 * loss_similarity

        loss = loss / loss.item()
        return loss
