# Cluster Fixed Points

import numpy as np
import torch

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
def find_clusters(tensor, eps=0.5, min_samples=5):
    # convert tensor to numpy for sklearn compatibility
    data = tensor.numpy()

    # apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    # get unique labels (clusters)
    unique_labels = np.unique(db.labels_)

    # calculate and collect the centroid for each cluster
    centroids = []
    for label in unique_labels:
        if label != -1:  # -1 label corresponds to noise in DBSCAN
            members = db.labels_ == label
            centroids.append(np.mean(data[members], axis=0))

    # convert list of centroids into a tensor
    centroids = torch.tensor(centroids)

    return centroids
