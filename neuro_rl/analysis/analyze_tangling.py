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
    
    scaler = sklearn.preprocessing.StandardScaler()
    X = scaler.fit_transform(X)

    dt = t[1] - t[0]

    t_diff = np.diff(t, axis=0)
    t_diff = np.insert(t_diff, 0, t_diff[0], axis=0)
    
    # compute derivatives of X
    X_dot = np.diff(X, axis=0) / dt
    X_dot = np.insert(X_dot, 0, X_dot[0,:], axis=0)

    # find first time step
    first_indices = np.where(t == 0)[0]
    last_indices = np.roll(first_indices, -1) - 1
    last_indices[-1] = len(t) - 1

    for i in range(len(first_indices)):
        X_dot[first_indices[i], :] = ( X[first_indices[i], :] - X[last_indices[i], :] ) / dt  # X_dot = X[first time step w/in CONDITION] - X[last time step w/in CONDITION] 
 
    # compute constant, prevents denominator from shrinking to zero
    epsilon = 0.1 * np.var(X)

    # get number of timesteps
    N_T = np.shape(X)[0]

    # get pairwise combinations of all timesteps
    # 2nd answer: https://stackoverflow.com/questions/464864/get-all-possible-2n-combinations-of-a-list-s-elements-of-any-length
    C = np.array(list(itertools.combinations(range(N_T), 2)))

    # initialize all arrays
    C_t = np.zeros((N_T,2))
    X_diff_t = np.zeros((N_T,1))
    X_dot_diff_t = np.zeros((N_T,1))
    Q = np.zeros((N_T,1), dtype=float)

    # iterate over each timestep, t
    for t in range(N_T):

        # get indices for all time-wise pair (specific t, all t')
        C_t = C[np.any(C==t, axis=1),:]

        # || x_dot(t) - x_dot(t') || ^2 for all (specific t, all t')
        X_dot_diff_t = np.sum ( np.square( X_dot[C_t[:,0],:] - X_dot[C_t[:,1],:] ) , axis=1)

        # || x(t) - x(t') || ^2 for all (specific t, all t')
        X_diff_t = np.sum ( np.square( X[C_t[:,0],:] - X[C_t[:,1],:] ) , axis=1)
        
        # compute Q(t)
        Q[t] = np.max ( X_dot_diff_t / ( X_diff_t + epsilon) )

    return Q
    
# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk

def export_tangling(pca: sklearn.decomposition.PCA, path: str):
    pk.dump(pca, open(path,"wb"))

def import_tangling(path: str):
    return pk.load(open(path,'rb'))

def analyze_tangling(path: str, data_names: List[str], file_suffix: str = ''):

    N_COMPONENTS = 10

    # load DataFrame
    data = pd.read_csv(path + 'RAW_DATA' + file_suffix + '.csv', index_col=0)

    # copy data DataFrame so can add tangling data to it
    data_w_tangling = data

    for idx, data_type in enumerate(data_names):

        # select data for PCA analysis
        filt_data = data.loc[:,data.columns.str.contains(data_type + '_RAW')].values
        time = data['TIME'].values

        if filt_data.shape[1] > 0:
                
            # computa pca
            tangling = compute_tangling(filt_data, time)

            tangling_df = pd.DataFrame(tangling, columns = [data_type + '_TANGLING'])

            # append tangling as a new column in the data DataFrame
            data_w_tangling = pd.concat([data_w_tangling, tangling_df], axis=1)

    # export DataFrame
    data_w_tangling.to_parquet(path + 'RAW_DATA' + file_suffix + '_WITH_TANGLING' + '.parquet')

    return data_w_tangling