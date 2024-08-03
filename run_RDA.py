import time
from methods.RDA import RDA
from utils.resample_loss import ResampleLoss
from utils.utils import set_seed
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from utils.metrics import evaluation_KLD
import argparse
from utils.metrics import evaluation_lt
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='sample_data')
parser.add_argument('--hidden_dim', type=int, default=100)
parser.add_argument('--lambda1', type=float, default=0.1)
parser.add_argument('--lambda2', type=float, default=0.1)
parser.add_argument('--lambda3', type=float, default=0.1)
parser.add_argument('--max_epoch', type=int, default=300)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--valid_size', type=int, default=20)
parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--seed', type=int, default=0)


def get_model():
    model = RDA(loss_func=ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                       focal=dict(focal=True, alpha=0.5, gamma=2),
                                       logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                       map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
                                       class_freq=np.sum(y_train, axis=0), train_num=x_train.shape[0],
                                       reduction='mean'),
                num_feature=x_train.shape[1],
                num_classes=y_train.shape[1],
                hidden_dim=hidden_dim,
                lambda1=lambda1,
                lambda2=lambda2,
                lambda3=lambda3,
                lr=lr,
                weight_decay=1e-4,
                adjust_lr=False,
                gradient_clip_value=5.0,
                max_epoch=max_epoch,
                verbose=False,
                device=device)
    return model


def _train():
    print('Start Training!')
    best_state_dict = None
    model = get_model()
    min_result = np.inf
    for epoch in range(max_epoch):
        model.train_loop(epoch=epoch, train_loader=train_loader)
        preds, ys = model.get_result(test_loader=val_loader)
        result = evaluation_KLD(ys, preds)
        if result < min_result:
            min_result = result
            best_state_dict = copy.deepcopy(model.state_dict())
    torch.save({'model': best_state_dict}, os.path.join(train_path, 'best.tar'))
    model.save(train_path, epoch=max_epoch - 1)


def _test():
    print('Start Testing!')
    model = get_model()
    model.load(train_path, epoch='best')
    preds, ys = model.get_result(test_loader=test_loader)
    return preds


if __name__ == '__main__':
    args = parser.parse_args()
    dataset = args.dataset
    hidden_dim = args.hidden_dim
    lambda1 = args.lambda1
    lambda2 = args.lambda2
    lambda3 = args.lambda3
    lr = args.lr
    max_epoch = args.max_epoch
    batch_size = args.batch_size
    valid_size = args.valid_size
    device = args.device
    seed = args.seed
    set_seed(seed)

    method = 'RDA'

    data = np.load(os.path.join('data', f'{dataset}.npz'))
    x_train = data['train_feature']
    x_test = data['test_feature']
    y_train = data['train_labels']
    y_test = data['test_labels']

    print(f'dataset: {dataset}, hidden_dim: {hidden_dim}, lambda1:{lambda1}, lambda2: {lambda2}, lambda3: {lambda3}')

    train_path = os.path.join('save', 'lt', f'{method}', 'train', f'{dataset}')
    result_path = os.path.join('save', 'lt', f'{method}')
    os.makedirs(train_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    if os.path.exists(os.path.join(result_path, f'{dataset}.npz')):
        print(method, dataset, 'exists!')
        result = np.load(f'save/lt/{method}/{dataset}.npz')
        y_pred = result['y_pred']
    else:
        print(method, dataset, 'training!')
        train_index, val_index = train_test_split(np.arange(x_train.shape[0]), test_size=valid_size,
                                                  shuffle=True, random_state=seed)
        x_val, y_val = x_train[val_index], y_train[val_index]
        x_train, y_train = x_train[train_index], y_train[train_index]

        train_dataset = TensorDataset(torch.from_numpy(x_train).float(), torch.from_numpy(y_train).float())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataset = TensorDataset(torch.from_numpy(x_val).float(), torch.from_numpy(y_val).float())
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_dataset = TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float())
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        model = get_model()

        # Training
        t = time.time()
        _train()
        training_time = time.time() - t
        # Predict
        t = time.time()
        y_pred = _test()
        test_time = time.time() - t

        np.savez(os.path.join(result_path, f'{dataset}.npz'),
                 y_pred=y_pred,
                 training_time=training_time,
                 test_time=test_time)

    result = evaluation_lt(y_test, y_pred, dataset)
    Chebyshev = result['Chebyshev']
    Clark = result['Clark']
    Canberra = result['Canberra']
    KLD = result['KLD']
    Cosine = result['Cosine']
    Intersection = result['Intersection']

    print('Chebyshev: %.4f' % Chebyshev)
    print('Clark: %.4f' % Clark)
    print('Canberra: %.4f' % Canberra)
    print('KLD: %.4f' % KLD)
    print('Cosine: %.4f' % Cosine)
    print('Intersection: %.4f' % Intersection)
