from pandas import read_csv
import numpy as np


def load_data(filename):
    """
    Load data from a csv file

    Parameters
    ----------
    filename : string
        Filename to be loaded.

    Returns
    -------
    X : ndarray
        the data matrix.

    y : ndarray
        the labels of each sample.
    """
    # this was just done
    data = read_csv(filename)
    z = np.array(data)
    y = z[:, 0]
    X = z[:, 1:]
    return X, y


def split_data(x, y, tr_fraction=0.5):
    """
    Split the data x, y into two random subsets

    """
    num_samples = x.size[1]
    num_training = int(num_samples * tr_fraction)
    num_testing = num_samples - num_training

    tr_index = np.zeros(num_samples, )
    tr_index[:num_training] = 1
    np.random.shuffle(tr_index)

    xtr = x[tr_index == 1, :]
    xts = x[tr_index == 0, :]

    ytr = y[tr_index == 1]
    yts = y[tr_index == 0]

    return xtr, ytr, xts, yts
