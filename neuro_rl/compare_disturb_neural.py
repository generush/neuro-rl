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

from analysis.compute_avg_gait_cycle import analyze_cycle
from analysis.append_pc import analyze_pca
from analysis.append_speed_axis import analyze_pca_speed_axis
from analysis.append_tangling import analyze_tangling
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


def plot_data(df, hc_perturb, obs_perturb, act_perturb, color, linestyle, axs_arr, ylabel_arr, scl, pca):
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
        # for j in range(np.shape(hc_perturb_pc[:,:,idx + IDX_BEGIN])[0]):
        #     axs_arr[0][idx].plot(time_idx, hc_perturb_pc[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color, linestyle=linestyle)
        axs_arr[0][idx].plot(time_idx, hc_perturb_avg_pc_df[column].values[0:tf], lw=2, c=color, linestyle=linestyle)
        # axs_arr[0][idx].set_title(f"{column}")
        axs_arr[0][idx].set_ylabel(ylabel_arr[0][idx])  # Set the ylabel here

    # BODY VELOCITY & ORIENTATION
    IDX_BEGIN = 0
    IDX_END = 12
    for idx, column in enumerate(obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
        # for j in range(np.shape(obs_perturb[:,:,idx + IDX_BEGIN])[0]):
        #     axs_arr[1][idx].plot(time_idx, obs_perturb[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color, linestyle=linestyle)
        axs_arr[1][idx].plot(time_idx, obs_perturb_avg_df[column].values[0:tf], lw=2, c=color, linestyle=linestyle)
        # axs_arr[1][idx].set_title(f"{column}")
        axs_arr[1][idx].set_ylabel(ylabel_arr[1][idx])  # Set the ylabel here

    # MOTOR TORQUES
    IDX_BEGIN = 0
    IDX_END = 12
    for idx, column in enumerate(act_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
        # for j in range(np.shape(act_perturb[:,:,idx + IDX_BEGIN])[0]):
        #     axs_arr[2][idx].plot(time_idx, act_perturb[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color, linestyle=linestyle)
        axs_arr[2][idx].plot(time_idx, act_perturb_avg_df[column].values[0:tf], lw=2, c=color, linestyle=linestyle)
        # axs_arr[1][idx].set_title(f"{column}")
        axs_arr[2][idx].set_ylabel(ylabel_arr[2][idx])  # Set the ylabel here

    # JOINT POSITION
    IDX_BEGIN = 12
    IDX_END = 24
    for idx, column in enumerate(obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
        # for j in range(np.shape(obs_perturb[:,:,idx + IDX_BEGIN])[0]):
        #     axs_arr[3][idx].plot(time_idx, obs_perturb[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color, linestyle=linestyle)
        axs_arr[3][idx].plot(time_idx, obs_perturb_avg_df[column].values[0:tf], lw=2, c=color, linestyle=linestyle)
        # axs_arr[2][idx].set_title(f"{column}")
        axs_arr[3][idx].set_ylabel(ylabel_arr[3][idx])  # Set the ylabel here

    # JOINT VELOCITY
    IDX_BEGIN = 24
    IDX_END = 36
    for idx, column in enumerate(obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
        # for j in range(np.shape(obs_perturb[:,:,idx + IDX_BEGIN])[0]):
        #     axs_arr[4][idx].plot(time_idx, obs_perturb[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color, linestyle=linestyle)
        axs_arr[4][idx].plot(time_idx, obs_perturb_avg_df[column].values[0:tf], lw=2, c=color, linestyle=linestyle)
        # axs_arr[3][idx].set_title(f"{column}")
        axs_arr[4][idx].set_ylabel(ylabel_arr[4][idx])  # Set the ylabel here

    # RATE OF CHANGE: BODY VELOCITY & ORIENTATION
    del_obs_perturb = np.diff(obs_perturb, axis=1)
    del_obs_perturb = np.pad(del_obs_perturb, ((0,0), (1,0), (0,0)), mode='constant')

    del_obs_perturb_avg_df = obs_perturb_avg_df.diff() / 0.02
    del_obs_perturb_avg_df = del_obs_perturb_avg_df.fillna(0)
    IDX_BEGIN = 0
    IDX_END = 12
    for idx, column in enumerate(del_obs_perturb_avg_df.columns[IDX_BEGIN:IDX_END]):
        # for j in range(np.shape(del_obs_perturb[:,:,idx + IDX_BEGIN])[0]):
        #     axs_arr[5][idx].plot(time_idx, del_obs_perturb[j,:L_TIME,idx + IDX_BEGIN], lw=1, c=color, linestyle=linestyle)
        axs_arr[5][idx].plot(time_idx, del_obs_perturb_avg_df[column].values[0:tf], lw=2, c=color, linestyle=linestyle)
        # axs_arr[4][idx].set_title(f"delta {column}")
        axs_arr[5][idx].set_ylabel(ylabel_arr[5][idx])  # Set the ylabel here



# SCL and PCA
NOM_DATA_PATHS = [
    '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_20-23-04_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]_LSTM4-DIST500-noperturb/',
    '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_20-19-14_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]_LSTM16-DIST500-noperturb/'
]

# SCL and PCA
NOM_DATA_PATHS = [
    '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_20-19-14_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]_LSTM16-DIST500-noperturb/',
    '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_20-19-14_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]_LSTM16-DIST500-noperturb/'
]

# +10% obs perturb
DIST_DATA_PATHS = [
    '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-08_16-49-59_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[256]_PC1-PC10_+25perturb_2.6s_one_step_LSTM16/',
    '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-08_18-18-49_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'
]


# load scaler and pca transforms
scls = [pk.load(open(path + 'A_LSTM_HC_SPEED_SCL.pkl','rb')) for path in NOM_DATA_PATHS]
pcas = [pk.load(open(path + 'A_LSTM_HC_SPEED_PCA.pkl','rb')) for path in NOM_DATA_PATHS]

# load dataframes
df_raws = [pd.read_parquet(path + 'RAW_DATA.parquet') for path in DIST_DATA_PATHS]

T_BEGIN = 2.1
T_END = 2.9
N_ENVS = 10

# For each unique environment
for ENV in range(N_ENVS):

    # Create the figures and axes for each dataframe
    fig, axs = plt.subplots(nrows=12, ncols=6, figsize=[18,14])
    axs_arr = axs.T  # Transpose the axes array so that the subplots are in 6 columns
    plt.subplots_adjust(hspace=0.5, wspace=0.75)

    for idx, df_raw in enumerate(df_raws):
        scl = scls[idx]
        pca = pcas[idx]
        
        color = 'red' if idx == 0 else 'black'  # Choose a color based on the dataframe
        linestyle='solid' if idx == 0 else 'dashed'  # Choose a color based on the dataframe

        figs = ['HC', 'OBS_BODY_VEL_ORIENT', 'ACT', 'OBS_JOINT_POS', 'OBS_JOINT_VEL', 'DELTA_OBS_BODY_VEL_ORIENT']

        for i in range(6):
            axs_arr[i][0].set_title(figs[i])

        df = df_raw.loc[(df_raw['TIME'] >= T_BEGIN) & (df_raw['TIME'] < T_END) & (df_raw['ENV'] == ENV)]

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
            ['PC 1','PC 2','PC 3','PC 4','PC 5','PC 6','PC 7','PC 8','PC 9','PC 10','PC 11','PC 12'],
            [r'$u$', r'$v$', r'$w$', r'$p$', r'$q$', r'$r$', r'$\phi$', r'$\theta$', r'$\psi$', r'$u*$', r'$v*$', r'$w*$'],
            ['','','','','','','','','','','',''],
            ['','','','','','','','','','','',''],
            ['','','','','','','','','','','',''],
            [r'$\dot{u}$', r'$\dot{v}$', r'$\dot{w}$', r'$\dot{p}$', r'$\dot{q}$', r'$\dot{r}$', r'$\dot{\phi}$', r'$\dot{\theta}$', r'$\dot{\psi}$', r'$\dot{u}*$', r'$\dot{v}*$', r'$\dot{w}*$'],
        ]

        # Plot the data
        plot_data(df, hc_perturb, obs_perturb, act_perturb, color, linestyle, axs_arr, ylabel_arr, scl, pca)

    # plot the background color
    for axs in axs_arr:
        for ax in axs:
            ax.fill_between([2.58, 2.6], ax.get_ylim()[0], ax.get_ylim()[1], color=mcolors.to_rgba('gray', alpha=0.2))  # adjust alpha for transparency

    SAVE_PATH = DIST_DATA_PATHS[0]
    FIGS = ['HC', 'OBS_BODY_VEL_ORIENT', 'ACT', 'OBS_JOINT_POS', 'OBS_JOINT_VEL', 'DELTA_OBS_BODY_VEL_ORIENT']

    filename = f'perturb_neural_ENV{ENV}'
    fig.savefig(SAVE_PATH + filename + '.pdf', format='pdf', dpi=600, facecolor=fig.get_facecolor())

    fig.show()

print('hi')




# FIRST LOOK AT OBS AND ACTIONS
# THEN LOOK AT HIDDEN STATES

# NEXT, PLOT THE TYPICAL ORBIT AND THEN PRINT THE PERTURBED ONE TOO.
# PLOT THE SAME FOR THE OTHER MODEL and for other disturbance levels. All on one.
# MOVE THIS TO A NEW SCRIPT
# THEN EXPORT THAT NEW DATA AND COMPUTE THE JACOBIANS AROUND THOSE OPERATING POINTS
# SEE IF THE LONGER LSTM MODEL HAS BETTER GRADIENTS ???


