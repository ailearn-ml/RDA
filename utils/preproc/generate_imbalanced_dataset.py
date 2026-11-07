import numpy as np
from sklearn.model_selection import KFold
import copy

values = {'Movie': [1, 2, 2, 0],
          'SCUT-FBP': [2, 0.1, 2, 4],
          'Emotion6': [0.8, 1, 3, 0],
          'Flickr_LDL': [0.8, 1, 2, 4],
          'RAF-ML': [5, 3.5, 2, 1],
          'Natural_Scene': [5, 2, 0, 4],
          }

for dataset in ['Movie', 'SCUT-FBP', 'Emotion6', 'Flickr_LDL', 'RAF-ML', 'Natural_Scene']:
    a, b, c, d = values[dataset]
    frequency = []
    data = np.load(f'../../data/dataset/{dataset}.npz')
    x = data['x']
    y = data['y']
    idx = KFold(n_splits=10, shuffle=True, random_state=0).split(np.arange(x.shape[0]))
    index = []
    for i in idx:
        y_train = copy.deepcopy(y)[i[0]]
        num_labels = y_train.shape[1]
        idx1 = set(np.arange(y_train.shape[0])[y_train[:, c] < a / num_labels].tolist())
        idx2 = set(np.arange(y_train.shape[0])[y_train[:, d] > b / num_labels].tolist())
        idx = idx1.intersection(idx2)
        for label in range(y_train.shape[1]):
            idx = idx.union(set(y_train[:, label].argsort()[::-1][:1].tolist()))
        idx = np.array(list(idx))

        train_idx = i[0][idx]
        y_train = y[train_idx]
        frequency.append(y_train.sum(0))
        MAX = np.max(y_train.sum(0))
        MIN = np.min(y_train.sum(0))
        index.append({'train_idx': train_idx, 'test_idx': i[1]})
    np.save(f'../../data/lt_idx/{dataset}_lt_idx.npy', index)
    frequency = np.array(frequency).mean(0)
    min_freq = np.min(frequency)
    few, many = [], []
    for i in range(y.shape[1]):
        if frequency[i] > min_freq * 10:
            many.append(i)
        else:
            few.append(i)

    print(few, many, frequency.max() / frequency.min())
