import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
import os
from collections import deque


def kaiming_normal_init_net(net):
    for name, param in net.named_parameters():
        if 'weight' in name and len(param.shape) == 2:
            nn.init.kaiming_normal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)


class LDLTemplate(nn.Module):

    def __init__(self, num_feature, num_classes, adjust_lr=False, gradient_clip_value=5.0,
                 max_epoch=None, verbose=False, device='cuda:0'):
        super(LDLTemplate, self).__init__()
        self.num_feature = num_feature
        self.num_classes = num_classes
        self.adjust_lr = adjust_lr
        self.epoch = 0
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.device = device

    @abstractmethod
    def set_forward(self, x):
        # x -> predicted score
        pass

    @abstractmethod
    def set_forward_loss(self, x, y):
        # batch -> loss value
        pass

    def train_loop(self, epoch, train_loader, log=None, print_freq=10):
        self.train()
        if not log:
            log = print
        self.epoch = epoch
        if self.adjust_lr:
            self.adjust_learning_rate()
        total_loss = 0
        num_samples = 0
        for i, batch in enumerate(train_loader):
            x = batch[0].to(self.device)
            y = batch[1].to(self.device)
            self.optimizer.zero_grad()
            loss = self.set_forward_loss(x, y)
            loss.backward()
            self.clip_gradient()
            self.optimizer.step()
            total_loss += loss.item()
            num_samples += x.shape[0]
            if self.verbose and (i % print_freq) == 0:
                log('Epoch %d | Batch %d/%d | Loss %f' % (
                    epoch, i, len(train_loader), total_loss / num_samples))
        avg_loss = total_loss / num_samples
        return avg_loss

    def get_result(self, test_loader):
        self.eval()
        preds = []
        ys = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                x = batch[0].to(self.device)
                y = batch[1].to(self.device)
                pred = self.set_forward(x).detach().cpu().numpy()
                preds.extend(pred)
                ys.extend(y.cpu().numpy())
        preds = np.array(preds)
        ys = np.array(ys)
        return preds, ys

    def save(self, path, epoch=None, save_optimizer=False):
        os.makedirs(path, exist_ok=True)
        if type(epoch) is str:
            save_path = os.path.join(path, '%s.tar' % epoch)
        elif epoch is None:
            save_path = os.path.join(path, 'model.tar')
        else:
            save_path = os.path.join(path, '%d.tar' % epoch)
        while True:
            try:
                if not save_optimizer:
                    torch.save({'model': self.state_dict(), }, save_path)
                else:
                    torch.save({'model': self.state_dict(),
                                'optimizer': self.optimizer.state_dict(), }, save_path)
                return
            except:
                pass

    def load(self, path, epoch=None, load_optimizer=False):
        if type(epoch) is str:
            load_path = os.path.join(path, '%s.tar' % epoch)
        else:
            if epoch is None:
                files = os.listdir(path)
                files = np.array(list(map(lambda x: int(x.replace('.tar', '')), files)))
                epoch = np.max(files)
            load_path = os.path.join(path, '%d.tar' % epoch)
        tmp = torch.load(load_path, map_location=self.device)
        self.load_state_dict(tmp['model'])
        if load_optimizer:
            self.optimizer.load_state_dict(tmp['optimizer'])

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))

    def adjust_learning_rate(self):
        epoch = self.epoch + 1
        if epoch <= 5:
            self.lr_now = self.lr * epoch / 5
        elif epoch >= int(self.max_epoch * 0.8):
            self.lr_now = self.lr * 0.01
        elif epoch > int(self.max_epoch * 0.6):
            self.lr_now = self.lr * 0.1
        else:
            self.lr_now = self.lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_now

    def cosine_similarity(self, x, y):
        '''
        Cosine Similarity of two tensors
        Args:
            x: torch.Tensor, m x d
            y: torch.Tensor, n x d
        Returns:
            result, m x n
        '''
        assert x.size(1) == y.size(1)
        x = torch.nn.functional.normalize(x, dim=1)
        y = torch.nn.functional.normalize(y, dim=1)
        return x @ y.transpose(0, 1)
