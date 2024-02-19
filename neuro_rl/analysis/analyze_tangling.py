# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

import numpy as np
import pandas as pd

from typing import List

from plotting.generation import generate_dropdown, generate_graph
from plotting.plot import plot_scatter3_ti_tf
from embeddings.embeddings import Data, Embeddings, MultiDimensionalScalingEmbedding, PCAEmbedding, MDSEmbedding, ISOMAPEmbedding,LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding

import sklearn.preprocessing
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

import itertools
import numpy as np

def compute_tangling(X: np.array, t: np.array):
    
    # scaler = sklearn.preprocessing.StandardScaler()
    # X = scaler.fit_transform(X)

    dt = t[1] - t[0]
    
    # compute derivatives of X
    X_dot = np.diff(X, axis=0) / dt
    X_dot = np.insert(X_dot, 0, X_dot[0,:], axis=0)

    # find first time step
    first_indices = np.where(t == 0)[0]
    last_indices = np.roll(first_indices, -1) - 1
    last_indices[-1] = len(t) - 1

    for i in range(len(first_indices)):
        X_dot[first_indices[i], :] = ( X[first_indices[i], :] - X[last_indices[i], :] ) / dt  # X_dot = X[first time step w/in CONDITION] - X[last time step w/in CONDITION] 
 
    # compute constant, prevents denominator from shrinking to 
    epsilon = 0.1 * np.var(X)

    # Calculate the pairwise squared differences for X and X_dot
    X_diff_t = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    X_dot_diff_t = np.sum((X_dot[:, None] - X_dot[None, :]) ** 2, axis=-1)
    
    # Calculate the ratios of X_dot_diff to X_diff
    ratios = X_dot_diff_t / ( X_diff_t + epsilon )
    # Find the maximum ratio
    Q = ratios.max(axis=0)

    return Q

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk

def export_tangling(pca: sklearn.decomposition.PCA, path: str):
    pk.dump(pca, open(path,"wb"))

def import_tangling(path: str):
    return pk.load(open(path,'rb'))

def analyze_tangling(df: pd.DataFrame, data_names: List[str], file_suffix: str = ''):

    # load DataFrame
    # data = pd.read_csv(path + 'PC_DATA' + file_suffix + '.csv', index_col=0)

    # copy data DataFrame so can add tangling data to it
    data_w_tangling = df

    for idx, data_type in enumerate(data_names):

        # select data for PCA analysis
        filt_data = df.loc[:,df.columns.str.contains(data_type + '_PC')].values
        time = df['TIME'].values

        # if filt_data.shape[1] > 0:
            
        # computa tangling
        tangling = compute_tangling(filt_data, time)

        tangling_df = pd.DataFrame(tangling, columns = [data_type + '_TANGLING'])

        # append tangling as a new column in the data DataFrame
        data_w_tangling = pd.concat([data_w_tangling, tangling_df], axis=1)

    # export DataFrame
    # data_w_tangling.to_parquet(path + 'RAW_DATA' + file_suffix + '_WITH_TANGLING' + '.parquet')
    # data_w_tangling.to_csv(path + 'RAW_DATA' + file_suffix + '_WITH_TANGLING' + '.csv')

    return data_w_tangling