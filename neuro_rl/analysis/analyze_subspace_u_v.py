# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

import numpy as np
import pandas as pd
import pickle as pk
import dask.dataframe as dd

def import_pca(path: str):
    return pk.load(open(path,'rb'))

FOLDER_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data_PAPER_test_subspace'

pca_u = import_pca(FOLDER_PATH + '/u[-1,-0.3,8]/ALSTM_CX_PCA.pkl')
pca_u2 = import_pca(FOLDER_PATH + '/u[-1,1,11]/ALSTM_CX_PCA.pkl')
pca_v = import_pca(FOLDER_PATH + '/v[-1.,-0.3,8]/ALSTM_CX_PCA.pkl')


W_U = pca_u.components_.transpose()
W_U2 = pca_u2.components_.transpose()
W_V = pca_v.components_.transpose()

# load DataFrame
df_u = dd.read_csv(FOLDER_PATH + '/u[-1,-0.3,8]/' + 'RAW_DATA_AVG' + '.csv')
df_u2 = dd.read_csv(FOLDER_PATH + '/u[-1,1,11]/' + 'RAW_DATA_AVG' + '.csv')
df_v = dd.read_csv(FOLDER_PATH + '/v[-1.,-0.3,8]/' + 'RAW_DATA_AVG' + '.csv')

# select data for PCA analysis (only raw data)
R_U = df_u.loc[:,df_u.columns.str.contains('ALSTM_CX' + '_RAW')].compute().to_numpy()
R_U2 = df_u2.loc[:,df_u2.columns.str.contains('ALSTM_CX' + '_RAW')].compute().to_numpy()
R_V = df_v.loc[:,df_v.columns.str.contains('ALSTM_CX' + '_RAW')].compute().to_numpy()

def V(R, W):
    return 1 - np.linalg.norm( R - np.matmul ( R, np.matmul( W, W.transpose() ) ) ) / np.linalg.norm(R)

subspace_overlap_U_V = V(R_V, W_U) / V(R_V, W_V)
subspace_overlap_U_U2 = V(R_U2, W_U) / V(R_U2, W_U2)

