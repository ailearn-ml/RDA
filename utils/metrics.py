import numpy as np


def positive(x):
    return np.maximum(x, 1e-14)


def chebyshev(true, pred):  # average Chebyshev distance
    return np.mean(np.max(np.abs(true - pred), axis=1))


def clark(true, pred):  # average Clark distance
    return np.mean(np.sqrt(np.sum((true - pred) ** 2 / positive((true + pred) ** 2), axis=1)))


def canberra(true, pred):  # average Canberra metric
    return np.mean(np.sum(np.abs(true - pred) / positive((true + pred)), axis=1))


def kld(true, pred):  # average Kullback-Leibler divergence
    return np.mean(np.sum(true * np.log(positive(true / positive(pred))), axis=1))


def cosine(true, pred):  # average Cosine coefficient
    return np.mean(np.sum(true * pred, axis=1) / positive(np.sum(true ** 2, axis=1) ** 0.5) / positive(
        np.sum(pred ** 2, axis=1) ** 0.5))


def intersection(true, pred):  # average intersection similarity
    return np.mean(np.sum(np.minimum(true, pred), axis=1))


def euclidean(true, pred):  # average Euclidean distance
    return np.mean(np.sum((true - pred) ** 2, axis=1) ** 0.5)


def evaluation_lt(true, pred, dataset_name):
    dataset_to_freq = {
        'sample_data': [[1, 2, 3, 4], [0]],
    }
    tail, head = dataset_to_freq[dataset_name]
    tail, head = np.array(tail), np.array(head)

    Chebyshev = chebyshev(true, pred)
    Clark = clark(true, pred)
    Canberra = canberra(true, pred)
    KLD = kld(true, pred)
    Cosine = cosine(true, pred)
    Intersection = intersection(true, pred)

    all = euclidean(true, pred)
    tail = euclidean(true[:, tail], pred[:, tail])
    head = euclidean(true[:, head], pred[:, head])

    return {'Chebyshev': Chebyshev, 'Clark': Clark, 'Canberra': Canberra,
            'KLD': KLD, 'Cosine': Cosine, 'Intersection': Intersection,
            'all': all, 'tail': tail, 'head': head}


def evaluation_KLD(true, pred):
    return kld(true, pred)
