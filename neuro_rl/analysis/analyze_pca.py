# https://plotly.com/python/3d-scatter-plots/

from typing import List

import numpy as np
import pandas as pd

import sklearn.preprocessing
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
from constants.normalization_types import NormalizationType
from utils.scalers import RangeScaler, RangeSoftScaler

def compute_pca(n_components, norm_type: NormalizationType):

    if norm_type == NormalizationType.Z_SCORE.value:
        # Perform z-score normalization
        scl = sklearn.preprocessing.StandardScaler()

    elif norm_type == NormalizationType.RANGE.value:
        # Perform range normalization
        scl = RangeScaler()
        
    elif norm_type == NormalizationType.RANGE_SOFT.value:
        # Perform range soft normalization
        scl = RangeSoftScaler(softening_factor=5)  # You can adjust the softening factor as needed
        
    else:

        raise ValueError(f"Unsupported normalization type: {norm_type}")
        
    # create PCA object
    pca = sklearn.decomposition.PCA(n_components=n_components)

    return scl, pca

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk

def export_scl(scl: sklearn.preprocessing.StandardScaler, path: str):
    pk.dump(scl, open(path,"wb"))

def export_pca(pca: sklearn.decomposition.PCA, path: str):
    pk.dump(pca, open(path,"wb"))

def import_scl(path: str):
    return pk.load(open(path,'rb'))

def import_pca(path: str):
    return pk.load(open(path,'rb'))

