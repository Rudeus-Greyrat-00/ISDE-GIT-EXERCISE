import numpy as np
from sklearn.metrics.pairwise import euclidean_distances, pairwise_distances


class NMC(object):
    """
    Class implementing the Nearest Mean Centroid (NMC) classifier.

    This classifier estimates one centroid per class from the training data,
    and predicts the label of a never-before-seen (test) point based on its
    closest centroid.

    Attributes
    -----------------
    - centroids: read-only attribute containing the centroid values estimated
        after training

    Methods
    -----------------
    - fit(x,y) estimates centroids from the training data
    - predict(x) predicts the class labels on testing points

    """

    def __init__(self):
        self._centroids = None
        self._class_labels = None  # class labels may not be contiguous indices

    @property
    def centroids(self):
        return self._centroids

    @property
    def class_labels(self):
        return self._class_labels

    def fit(self, xtr, ytr):
        n_classes = np.unique(ytr).size
        n_features = xtr.shape[1]

        centroids = np.zeros(shape=(n_classes, n_features),)

        for i in range(0, n_classes):
            i_s = xtr[ytr == i, :]
            centroid = np.mean(i_s, axis=0)
            centroids[i, :] = centroid

        self._centroids = centroids

    def predict(self, xts):
        if self._centroids is None:
            raise ValueError("NMC is not fit")

        distances = pairwise_distances(xts, self._centroids)

        y = np.argmin(distances, axis=1)

        return y
