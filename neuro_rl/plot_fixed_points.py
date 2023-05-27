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

import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

import time

import torch


# no bias
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-26_09-57-04_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'

# no bias but pos u and neg u
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_08-29-41_u[-1,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_10-29-52_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

import h5py


# Load the HDF5 file
with h5py.File(DATA_PATH + 'cx_traj.h5', 'r') as f:
    # Read the dataset from the file
    cx_traj = f['cx_traj'][:]
    # Convert NumPy array to PyTorch tensor
    cx_traj = torch.from_numpy(cx_traj)


# Load the HDF5 file
with h5py.File(DATA_PATH + 'hx_traj.h5', 'r') as f:
    # Read the dataset from the file
    hx_traj = f['hx_traj'][:]
    # Convert NumPy array to PyTorch tensor
    hx_traj = torch.from_numpy(hx_traj)


# Load the HDF5 file
with h5py.File(DATA_PATH + 'q_traj.h5', 'r') as f:
    # Read the dataset from the file
    q_traj = f['q_traj'][:]
    # Convert NumPy array to PyTorch tensor
    q_traj = torch.from_numpy(q_traj)

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk
pca = pk.load(open(DATA_PATH + 'info_A_LSTM_CX_PCA.pkl','rb'))
scl = pk.load(open(DATA_PATH + 'A_LSTM_CX_SPEED_SCL.pkl','rb'))

cycle_x = pd.read_csv(DATA_PATH + 'info_A_LSTM_CX_x_by_speed.csv', index_col=0)
cycle_y = pd.read_csv(DATA_PATH + 'info_A_LSTM_CX_y_by_speed.csv', index_col=0)
cycle_z = pd.read_csv(DATA_PATH + 'info_A_LSTM_CX_z1_by_speed.csv', index_col=0)

cycle_x = cycle_x.to_numpy().reshape(-1)
cycle_y = cycle_y.to_numpy().reshape(-1)
cycle_z = cycle_z.to_numpy().reshape(-1)



M = 4096
EPOCHS = 10000
SAMPLE_RATE = 100

cx_pc = np.zeros((EPOCHS//SAMPLE_RATE,M,12))
cx_pc_end = np.zeros((M,12))

cx_pc_end = pca.transform(scl.transform(torch.squeeze(cx_traj[-1,:,:]).detach().cpu().numpy()))
cx_q_end = torch.squeeze(q_traj[-1,:]).detach().cpu().numpy()

for i in range(EPOCHS//SAMPLE_RATE):
    cx_pc[i,:,:] = pca.transform(scl.transform(torch.squeeze(cx_traj[i,:,:]).detach().cpu().numpy()))

import numpy as np
import matplotlib.pyplot as plt

# Reshape the array to have a single dimension for scatter plotting
cx_pc_reshaped = cx_pc[:,:,:3].reshape(-1, 3)
cx_q = q_traj[:,:].detach().cpu().numpy().reshape(-1)

# Extract x, y, and z coordinates
cx_x = cx_pc_reshaped[:, 0]
cx_y = cx_pc_reshaped[:, 1]
cx_z = cx_pc_reshaped[:, 2]

# Create the scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(cx_x, cx_y, cx_z, c=np.log10(cx_q), cmap='viridis', s=1)
ax.scatter(cx_pc_end[:,0], cx_pc_end[:,1], cx_pc_end[:,2], c=np.log10(cx_q_end), cmap='viridis', s=10)
ax.scatter(cycle_x, cycle_y, cycle_z, s=1, c='r')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot')

# Add a colorbar to the plot
cbar = plt.colorbar(ax.collections[0])
cbar.set_label('q_reshaped')

# Show the plot
plt.show()

print('hi')