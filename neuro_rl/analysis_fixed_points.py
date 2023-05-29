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

model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_04-17-35-59/nn/last_AnymalTerrain_ep_2950_rew_20.14143.pth')

# no bias
model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_25-14-47-18/nn/last_AnymalTerrain_ep_2950_rew_20.2923.pth')

# no bias but pos u and neg u, no noise/perturb
model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_27-12-07-49/nn/last_AnymalTerrain_ep_1800_rew_21.021248.pth')



import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.autograd import grad
import h5py

INPUT_DIM = 256
HIDDEN_DIM = 128
N_LAYERS = 1
OUTPUT_DIM = 128
M = 4096
SAMPLE_RATE = 100
MAX_ITERATIONS = 10000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model and LSTM
# model = torch.load('model_path', map_location=device)
a_rnn = nn.LSTM(INPUT_DIM, HIDDEN_DIM, N_LAYERS).to(device)
state_dict = {key.replace('a2c_network.a_rnn.rnn.', ''): value for key, value in model['model'].items() if key.startswith('a2c_network.a_rnn.rnn')}
a_rnn.load_state_dict(state_dict)

# Prepare input data
input_data = torch.zeros(1, M, INPUT_DIM, dtype=torch.float32).to(device)
random_numbers = [[random.random() for _ in range(HIDDEN_DIM * 2)] for _ in range(M//2)]
h1 = 1 * torch.tensor(random_numbers, dtype=torch.float32).reshape(1, M//2, HIDDEN_DIM * 2).to(device)
h2 = 5 * torch.tensor(random_numbers, dtype=torch.float32).reshape(1, M//2, HIDDEN_DIM * 2).to(device)
h = torch.cat((h1, h2), dim=1)
h.requires_grad = True

# Other data
gamma = 0.1 * torch.ones(1, M, 1, dtype=torch.float32).to(device)
hx_traj = torch.zeros(MAX_ITERATIONS//SAMPLE_RATE, M, HIDDEN_DIM, dtype=torch.float32)
cx_traj = torch.zeros(MAX_ITERATIONS//SAMPLE_RATE, M, HIDDEN_DIM, dtype=torch.float32)
q_traj = torch.zeros(MAX_ITERATIONS//SAMPLE_RATE, M, dtype=torch.float32)

q = torch.zeros(1, M, dtype=torch.float32).to(device)
q_last = torch.zeros(1, M, dtype=torch.float32).to(device)
q_last_last = torch.zeros(1, M, dtype=torch.float32).to(device)

for epoch in range(MAX_ITERATIONS):
    a_rnn_output, (a_rnn_hx, a_rnn_cx) = a_rnn(input_data, (h[:,:,:HIDDEN_DIM].contiguous(), h[:,:,HIDDEN_DIM:].contiguous()))
    _h = torch.cat((a_rnn_hx, a_rnn_cx), dim=2)
    q = torch.norm(h - _h, dim=2)

    if epoch % 100 == 0:
        # print only every 100 epochs
        min_index = torch.argmin(torch.norm(q, dim=0))
        print(f"epoch: {epoch}, _h min idx|: {min_index.item()}, |_h|: {torch.norm(h[:,min_index,:]).item():.2e}, q: {q[:,min_index].item():.2e}, |q - q_last|: {torch.norm(q[:,min_index] - q_last[:,min_index]).item():.2e}, gamma: {gamma[:, min_index, :].item():.2e}, |q - q_last_last|: {torch.norm(q[:,min_index] - q_last_last[:,min_index]).item():.2e}")

    gradient = torch.ones_like(q)
    q.backward(gradient)

    h = h.detach() - gamma * h.grad
    h.requires_grad = True

    mask = torch.norm(q - q_last, dim=0) > 10 * torch.norm(q - q_last_last, dim=0)
    gamma *= torch.where(mask.unsqueeze(dim=1).unsqueeze(dim=0), 0.999, 1.0)

    q_last_last = q_last
    q_last = q

    if epoch % SAMPLE_RATE == 0:
        hx_traj[epoch//SAMPLE_RATE,:,:] = h[:,:,:HIDDEN_DIM].cpu()
        cx_traj[epoch//SAMPLE_RATE,:,:] = h[:,:,HIDDEN_DIM:].cpu()
        q_traj[epoch//SAMPLE_RATE,:] = q[:,:].cpu()

DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-26_09-57-04_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_08-29-41_u[-1,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_10-29-52_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# no bias but pos u and neg u, no noise/perturb
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_17-11-41_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# Save data to hdf5
with h5py.File(DATA_PATH + 'cx_traj.h5', 'w') as f:
    f.create_dataset('cx_traj', data=cx_traj.detach().numpy())
with h5py.File(DATA_PATH + 'hx_traj.h5', 'w') as f:
    f.create_dataset('hx_traj', data=hx_traj.detach().numpy())
with h5py.File(DATA_PATH + 'q_traj.h5', 'w') as f:
    f.create_dataset('q_traj', data=q_traj.detach().numpy())




# PLOT EVOLUTION WITH ZERO INPUT

data = pd.read_csv(DATA_PATH + 'RAW_DATA_AVG.csv')
N_ENTRIES = len(data)
input = torch.zeros((1, N_ENTRIES, INPUT_DIM), device=device,  dtype=torch.float32)
hx = torch.tensor(data.loc[:,data.columns.str.contains('A_LSTM_HX')].values, device=device,  dtype=torch.float32).unsqueeze(dim=0)
cx = torch.tensor(data.loc[:,data.columns.str.contains('A_LSTM_CX')].values, device=device,  dtype=torch.float32).unsqueeze(dim=0)

hx_out = torch.zeros_like(hx)
cx_out = torch.zeros_like(cx)

a_rnn_out, (hx, cx) = a_rnn(input, (hx.contiguous(), cx.contiguous()))

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk
pca = pk.load(open(DATA_PATH + 'info_A_LSTM_CX_PCA.pkl','rb'))
scl = pk.load(open(DATA_PATH + 'A_LSTM_CX_SPEED_SCL.pkl','rb'))

cx_pc = pca.transform(scl.transform(torch.squeeze(cx).detach().cpu().numpy()))
cx_out_pc = pca.transform(scl.transform(torch.squeeze(cx_out).detach().cpu().numpy()))



import numpy as np
import matplotlib.pyplot as plt

# Generate some random 3D arrays
array1 = np.random.random((10, 10, 10))
array2 = np.random.random((10, 10, 10))

# Create a figure and subplots
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# Plot the first set of 3D arrays
x1, y1, z1 = np.meshgrid(np.arange(array1.shape[0]), np.arange(array1.shape[1]), np.arange(array1.shape[2]))
ax1.scatter(cx_pc[1:,0], cx_pc[1:,1], cx_pc[1:,2], c='r', cmap='viridis')

# Plot the second set of 3D arrays
x2, y2, z2 = np.meshgrid(np.arange(array2.shape[0]), np.arange(array2.shape[1]), np.arange(array2.shape[2]))
ax1.scatter(cx_out_pc[1:,0], cx_out_pc[1:,1], cx_out_pc[1:,2], c='b', cmap='viridis')


# Plot the first set of 3D arrays
x1, y1, z1 = np.meshgrid(np.arange(array1.shape[0]), np.arange(array1.shape[1]), np.arange(array1.shape[2]))
ax1.scatter(cx_pc[0,0], cx_pc[0,1], cx_pc[0,2], c='g', cmap='viridis')

# Plot the second set of 3D arrays
x2, y2, z2 = np.meshgrid(np.arange(array2.shape[0]), np.arange(array2.shape[1]), np.arange(array2.shape[2]))
ax1.scatter(cx_out_pc[0,0], cx_out_pc[0,1], cx_out_pc[0,2], c='k', cmap='viridis')


# Set labels and title for the subplots
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
ax1.set_title('Array 1')

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title('Array 2')

# Adjust subplot spacing
fig.tight_layout()

# Show the plot
plt.show()



print('Finished processing.')






















