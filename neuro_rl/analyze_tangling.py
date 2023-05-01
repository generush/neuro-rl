# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

import numpy as np
import pandas as pd

from utils.data_processing import process_data
from plotting.generation import generate_dropdown, generate_graph
from plotting.plot import plot_scatter3_ti_tf
from embeddings.embeddings import Data, Embeddings, MultiDimensionalScalingEmbedding, PCAEmbedding, MDSEmbedding, ISOMAPEmbedding,LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding

import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

######################## Loading all data files ##########################

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/'

# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/shadowhand_2023_03_11_1279/'
# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/anymalterrain_2023_04_17_00/'
# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/anymalterrain_2023_04_17_01/'
# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/anymalterrain_2023_04_17_AGENT_17_44/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data_AnymalTerrain_Flat_t0_t1000/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/'

import itertools
import numpy as np

def compute_tangling(X: np.array, dt: float):
    
    # compute derivatives of X
    X_dot = np.diff(X, axis=0) / dt
    X_dot = np.insert(X_dot, 0, (X[0,:] - X[-1,:] ) / dt, 0)

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


N_COMPONENTS = 10

dt = 0.005

# load DataFrame
data = process_data(DATA_PATH + 'RAW_DATA' + '.csv')

DATASETS = [
    # 'OBS',
    # 'ACT',
    'AHX',
    # 'CHX'
]

for idx, data_type in enumerate(DATASETS):

    # select data for PCA analysis
    filt_data = data.loc[:,data.columns.str.contains(data_type + '_RAW')].compute()

    # computa pca
    pca, pc_df = compute_tangling(filt_data, N_COMPONENTS, COLUMNS)

    # export PCA object
    export_tangling(pca, DATA_PATH + data_type +'_PCA' + '.pkl')

    # export DataFrame
    pc_df.insert(loc=0, column='TIME', value=data['TIME'].compute())
    pc_df.set_index('TIME', inplace=True)
    pc_df.to_csv(DATA_PATH + data_type + '_PC_DATA' + '.csv')

