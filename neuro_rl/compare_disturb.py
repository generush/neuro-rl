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

from analysis.analyze_cycle import analyze_cycle
from analysis.analyze_pca import analyze_pca
from analysis.analyze_pca_speed_axis import analyze_pca_speed_axis
from analysis.analyze_tangling import analyze_tangling
from plotting.dashboard import run_dashboard

from analysis.cluster import find_clusters
from analysis.jacobian import compute_jacobian, compute_jacobian_alternate, compute_jacobian_alternate2

import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

import time

import torch

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import random

import h5py

import numpy as np
import pandas as pd

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt

HIDDEN_SIZE = 128

NOM_DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_20-19-14_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]_LSTM16-DIST500-noperturb/'

# load scaler and pca transforms
scl = pk.load(open(NOM_DATA_PATH + 'A_LSTM_HC_SPEED_SCL.pkl','rb'))
pca = pk.load(open(NOM_DATA_PATH + 'A_LSTM_HC_SPEED_PCA.pkl','rb'))


DIST_DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_20-05-52_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]_LSTM16-DIST500-minus2500v/'

# AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM4-DIST500) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=4, horizon_length=16) (w/o bias) DIST1000
df_raw = pd.read_parquet(DIST_DATA_PATH + 'RAW_DATA.parquet')

T_BEGIN = 0
T_END = 4

df = df_raw.loc[(df_raw['TIME'] >= T_BEGIN) & (df_raw['TIME'] < T_END)]

hc_perturb_df = df.loc[:, df.columns.str.contains('A_LSTM_HC_RAW')]
obs_perturb_df = df.loc[:, df.columns.str.contains('OBS_RAW')]

# Reshape HC
time_length = len(df['TIME'].unique())
env_length = len(df['ENV'].unique())
feature_length = np.array(hc_perturb_df).shape[1]
hc_perturb = np.array(hc_perturb_df).reshape(env_length, time_length, feature_length)

# Reshape OBS
time_length = len(df['TIME'].unique())
env_length = len(df['ENV'].unique())
feature_length = np.array(obs_perturb_df).shape[1]
obs_perturb = np.array(obs_perturb_df).reshape(env_length, time_length, feature_length)

# Assuming your data is in a DataFrame called 'df'
averaged_data = df.groupby('TIME').mean()
averaged_data.reset_index(inplace=True)

# Set 'TIME' as the index
averaged_data.set_index('TIME', inplace=True)

hc_perturb_avg_df = averaged_data.loc[:, averaged_data.columns.str.contains('A_LSTM_HC_RAW')]
obs_perturb_avg_df = averaged_data.loc[:, averaged_data.columns.str.contains('OBS_RAW')]

hc_perturb_avg = torch.tensor(hc_perturb_avg_df.to_numpy())
hc_perturb_avg_pc = pca.transform(scl.transform(torch.squeeze(hc_perturb_avg).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy())).reshape(hc_perturb_avg.shape)
hc_perturb_avg_pc_df = pd.DataFrame(hc_perturb_avg_pc)
hc_perturb_pc = pca.transform(scl.transform(torch.squeeze( torch.tensor(hc_perturb)).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy())).reshape(hc_perturb.shape)

TIME_IDX = len(averaged_data)
ENV = df['ENV'].nunique()

time_idx = averaged_data.index.values

# HC
fig, axs = plt.subplots(nrows=12, ncols=1)
axs = axs.flatten()
IDX_BEGIN = 0
IDX_END = 12
for idx, column in enumerate(hc_perturb_avg_pc_df.columns[IDX_BEGIN:IDX_END]):
    for j in range(np.shape(hc_perturb_pc[:,:,idx + IDX_BEGIN])[0]):
        axs[idx].plot(time_idx, hc_perturb_pc[j,:,idx + IDX_BEGIN], lw=1, c='gray')
    axs[idx].plot(time_idx, hc_perturb_avg_pc_df[column].values, lw=2, c='k')
    axs[idx].set_title(f"{column}")
    

# BODY VELOCITY & ORIENTATION
fig, axs = plt.subplots(nrows=12, ncols=1)
axs = axs.flatten()
IDX_BEGIN = 0
IDX_END = 12
for idx, column in enumerate(obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
    for j in range(np.shape(obs_perturb[:,:,idx + IDX_BEGIN])[0]):
        axs[idx].plot(time_idx, obs_perturb[j,:,idx + IDX_BEGIN], lw=1, c='gray')
    axs[idx].plot(time_idx, obs_perturb_avg_df[column].values, lw=2, c='k')
    axs[idx].set_title(f"{column}")

# JOINT POSITION
fig, axs = plt.subplots(nrows=12, ncols=1)
axs = axs.flatten()
IDX_BEGIN = 12
IDX_END = 24
for idx, column in enumerate(obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
    for j in range(np.shape(obs_perturb[:,:,idx + IDX_BEGIN])[0]):
        axs[idx].plot(time_idx, obs_perturb[j,:,idx + IDX_BEGIN], lw=1, c='gray')
    axs[idx].plot(time_idx, obs_perturb_avg_df[column].values, lw=2, c='k')
    axs[idx].set_title(f"{column}")


# JOINT VELOCITY
fig, axs = plt.subplots(nrows=12, ncols=1)
axs = axs.flatten()
IDX_BEGIN = 24
IDX_END = 36
for idx, column in enumerate(obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
    for j in range(np.shape(obs_perturb[:,:,idx + IDX_BEGIN])[0]):
        axs[idx].plot(time_idx, obs_perturb[j,:,idx + IDX_BEGIN], lw=1, c='gray')
    axs[idx].plot(time_idx, obs_perturb_avg_df[column].values, lw=2, c='k')
    axs[idx].set_title(f"{column}")




plt.show()


        

# FIRST LOOK AT OBS AND ACTIONS
# THEN LOOK AT HIDDEN STATES

# NEXT, PLOT THE TYPICAL ORBIT AND THEN PRINT THE PERTURBED ONE TOO.
# PLOT THE SAME FOR THE OTHER MODEL and for other disturbance levels. All on one.
# MOVE THIS TO A NEW SCRIPT
# THEN EXPORT THAT NEW DATA AND COMPUTE THE JACOBIANS AROUND THOSE OPERATING POINTS
# SEE IF THE LONGER LSTM MODEL HAS BETTER GRADIENTS ???


