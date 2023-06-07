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
import matplotlib.colors as mcolors

HIDDEN_SIZE = 128


def plot_data(df, hc_perturb, obs_perturb, act_perturb, color, axs_arr, ylabel_arr, scl, pca):
    averaged_data = df.groupby('TIME').mean()
    averaged_data.reset_index(inplace=True)
    averaged_data.set_index('TIME', inplace=True)

    hc_perturb_avg_df = averaged_data.loc[:, averaged_data.columns.str.contains('A_LSTM_HC_RAW')]
    obs_perturb_avg_df = averaged_data.loc[:, averaged_data.columns.str.contains('OBS_RAW')]
    act_perturb_avg_df = averaged_data.loc[:, averaged_data.columns.str.contains('ACT_RAW')]

    hc_perturb_avg = torch.tensor(hc_perturb_avg_df.to_numpy())
    hc_perturb_avg_pc = pca.transform(scl.transform(torch.squeeze(hc_perturb_avg).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy())).reshape(hc_perturb_avg.shape)
    hc_perturb_avg_pc_df = pd.DataFrame(hc_perturb_avg_pc)
    hc_perturb_pc = pca.transform(scl.transform(torch.squeeze( torch.tensor(hc_perturb)).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy())).reshape(hc_perturb.shape)

    time_idx = averaged_data.index.values

    # is dataset short (contains zeros)?
    # Find the first zero element from the start
    zero_indices = np.where(obs_perturb[:,:,3] == 0)[1]

    # Check if there are any zero indices
    L_TIME = len(time_idx)
    tf = L_TIME
    if zero_indices.size > 0:
        tf = zero_indices[0]
    else:
        tf = None

    time_idx = time_idx[0:tf]
    L_TIME = len(time_idx)
    
    # HC
    IDX_BEGIN = 0
    IDX_END = 12
    for idx, column in enumerate(hc_perturb_avg_pc_df.columns[IDX_BEGIN:IDX_END]):
        for j in range(np.shape(hc_perturb_pc[:,:,idx + IDX_BEGIN])[0]):
            axs_arr[0][idx].plot(time_idx, hc_perturb_pc[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color)
        axs_arr[0][idx].plot(time_idx, hc_perturb_avg_pc_df[column].values[0:tf], lw=2, c=color)
        # axs_arr[0][idx].set_title(f"{column}")
        axs_arr[0][idx].set_ylabel(ylabel_arr[0][idx])  # Set the ylabel here

    # BODY VELOCITY & ORIENTATION
    IDX_BEGIN = 0
    IDX_END = 12
    for idx, column in enumerate(obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
        for j in range(np.shape(obs_perturb[:,:,idx + IDX_BEGIN])[0]):
            axs_arr[1][idx].plot(time_idx, obs_perturb[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color)
        axs_arr[1][idx].plot(time_idx, obs_perturb_avg_df[column].values[0:tf], lw=2, c=color)
        # axs_arr[1][idx].set_title(f"{column}")
        axs_arr[1][idx].set_ylabel(ylabel_arr[1][idx])  # Set the ylabel here

    # MOTOR TORQUES
    IDX_BEGIN = 0
    IDX_END = 12
    for idx, column in enumerate(act_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
        for j in range(np.shape(act_perturb[:,:,idx + IDX_BEGIN])[0]):
            axs_arr[2][idx].plot(time_idx, act_perturb[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color)
        axs_arr[2][idx].plot(time_idx, act_perturb_avg_df[column].values[0:tf], lw=2, c=color)
        # axs_arr[1][idx].set_title(f"{column}")
        axs_arr[2][idx].set_ylabel(ylabel_arr[2][idx])  # Set the ylabel here

    # JOINT POSITION
    IDX_BEGIN = 12
    IDX_END = 24
    for idx, column in enumerate(obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
        for j in range(np.shape(obs_perturb[:,:,idx + IDX_BEGIN])[0]):
            axs_arr[3][idx].plot(time_idx, obs_perturb[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color)
        axs_arr[3][idx].plot(time_idx, obs_perturb_avg_df[column].values[0:tf], lw=2, c=color)
        # axs_arr[2][idx].set_title(f"{column}")
        axs_arr[3][idx].set_ylabel(ylabel_arr[3][idx])  # Set the ylabel here

    # JOINT VELOCITY
    IDX_BEGIN = 24
    IDX_END = 36
    for idx, column in enumerate(obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
        for j in range(np.shape(obs_perturb[:,:,idx + IDX_BEGIN])[0]):
            axs_arr[4][idx].plot(time_idx, obs_perturb[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color)
        axs_arr[4][idx].plot(time_idx, obs_perturb_avg_df[column].values[0:tf], lw=2, c=color)
        # axs_arr[3][idx].set_title(f"{column}")
        axs_arr[4][idx].set_ylabel(ylabel_arr[4][idx])  # Set the ylabel here

    # RATE OF CHANGE: BODY VELOCITY & ORIENTATION
    del_obs_perturb = np.diff(obs_perturb, axis=1)
    del_obs_perturb = np.pad(del_obs_perturb, ((0,0), (1,0), (0,0)), mode='constant')

    del_obs_perturb_avg_df = obs_perturb_avg_df.diff() / 0.02
    del_obs_perturb_avg_df = del_obs_perturb_avg_df.fillna(0) / 0.02
    IDX_BEGIN = 0
    IDX_END = 12
    for idx, column in enumerate(del_obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
        for j in range(np.shape(del_obs_perturb[:,:,idx + IDX_BEGIN])[0]):
            axs_arr[5][idx].plot(time_idx, del_obs_perturb[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color)
        axs_arr[5][idx].plot(time_idx, del_obs_perturb_avg_df[column].values[0:tf], lw=2, c=color)
        # axs_arr[4][idx].set_title(f"delta {column}")
        axs_arr[5][idx].set_ylabel(ylabel_arr[5][idx])  # Set the ylabel here




# Create the figures and axes for each dataframe
figs, axs_arr = [], []
for _ in range(6):
    fig, axs = plt.subplots(nrows=12, ncols=1, figsize=[4,15])
    figs.append(fig)
    axs_arr.append(axs.flatten())

# Load the dataframes

NOM_DATA_PATHS = [
    '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_20-23-04_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]_LSTM4-DIST500-noperturb/',
    '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_20-19-14_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]_LSTM16-DIST500-noperturb/'
]

DIST_DATA_PATHS = [
    '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-07_00-22-38_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]]_LSTM4_DIST500-minus2500v/',
    '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_20-05-52_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]_LSTM16-DIST500-minus2500v/'
]

# load scaler and pca transforms
scls = [pk.load(open(path + 'A_LSTM_HC_SPEED_SCL.pkl','rb')) for path in NOM_DATA_PATHS]
pcas = [pk.load(open(path + 'A_LSTM_HC_SPEED_PCA.pkl','rb')) for path in NOM_DATA_PATHS]

# load dataframes
df_raws = [pd.read_parquet(path + 'RAW_DATA.parquet') for path in DIST_DATA_PATHS]

# For each dataframe
for idx, df_raw in enumerate(df_raws):
    scl = scls[idx]
    pca = pcas[idx]
    
    color_light = 'pink' if idx == 0 else 'gray'  # Choose a color based on the dataframe
    color = 'red' if idx == 0 else 'black'  # Choose a color based on the dataframe

    T_BEGIN = 0
    T_END = 4
    N_ENVS = 1

    df = df_raw.loc[(df_raw['TIME'] >= T_BEGIN) & (df_raw['TIME'] < T_END) & (df_raw['ENV'] < N_ENVS)]

    hc_perturb_df = df.loc[:, df.columns.str.contains('A_LSTM_HC_RAW')]
    obs_perturb_df = df.loc[:, df.columns.str.contains('OBS_RAW')]
    act_perturb_df = df.loc[:, df.columns.str.contains('ACT_RAW')]

    # Reshape HC
    env_length = len(df['ENV'].unique())
    time_length = len(df['TIME'].unique())
    feature_length = np.array(hc_perturb_df).shape[1]
    hc_perturb = np.array(hc_perturb_df).reshape(env_length, time_length, feature_length)

    # Reshape OBS
    env_length = len(df['ENV'].unique())
    time_length = len(df['TIME'].unique())
    feature_length = np.array(obs_perturb_df).shape[1]
    obs_perturb = np.array(obs_perturb_df).reshape(env_length, time_length, feature_length)

    # Reshape OBS
    env_length = len(df['ENV'].unique())
    time_length = len(df['TIME'].unique())
    feature_length = np.array(act_perturb_df).shape[1]
    act_perturb = np.array(act_perturb_df).reshape(env_length, time_length, feature_length)

    ylabel_arr = [
        ['','','','','','','','','','','',''],
        [r'$u$', r'$v$', r'$w$', r'$p$', r'$q$', r'$r$', r'$\phi$', r'$\theta$', r'$\psi$', r'$u*$', r'$v*$', r'$w*$'],
        ['','','','','','','','','','','',''],
        ['','','','','','','','','','','',''],
        ['','','','','','','','','','','',''],
        [r'$\dot{u}$', r'$\dot{v}$', r'$\dot{w}$', r'$\dot{p}$', r'$\dot{q}$', r'$\dot{r}$', r'$\dot{\phi}$', r'$\dot{\theta}$', r'$\dot{\psi}$', r'$\dot{u}*$', r'$\dot{v}*$', r'$\dot{w}*$'],
    ]

    # Plot the data
    plot_data(df, hc_perturb, obs_perturb, act_perturb, color, axs_arr, ylabel_arr, scl, pca)

# plot the background color
for axs in axs_arr:
    for ax in axs:
        ax.fill_between([2, 2.25], ax.get_ylim()[0], ax.get_ylim()[1], color=mcolors.to_rgba('gray', alpha=0.2))  # adjust alpha for transparency

SAVE_PATH = ['/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/']
FIGS = ['HC', 'OBS_BODY_VEL_ORIENT', 'OBS_BODY', 'ACT', 'OBS_JOINT_POS', 'OBS_JOINT_VEL', 'DELTA_OBS_BODY_VEL_ORIENT']
i = 0
# Show the figures
for fig in figs:
    fig.tight_layout()  # Add this line here

    filename = f"{FIGS[i]}"
    fig.savefig(SAVE_PATH + filename + '.pdf', format='pdf', dpi=600, facecolor=fig.get_facecolor())

    fig.show()

    i+=1

print('hi')




# FIRST LOOK AT OBS AND ACTIONS
# THEN LOOK AT HIDDEN STATES

# NEXT, PLOT THE TYPICAL ORBIT AND THEN PRINT THE PERTURBED ONE TOO.
# PLOT THE SAME FOR THE OTHER MODEL and for other disturbance levels. All on one.
# MOVE THIS TO A NEW SCRIPT
# THEN EXPORT THAT NEW DATA AND COMPUTE THE JACOBIANS AROUND THOSE OPERATING POINTS
# SEE IF THE LONGER LSTM MODEL HAS BETTER GRADIENTS ???


