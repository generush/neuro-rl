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

model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_29-22-48-36/nn/last_AnymalTerrain_ep_4700_rew_20.763342.pth')

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
MAX_ITERATIONS = 20000

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load model and LSTM
# model = torch.load('model_path', map_location=device)
a_rnn = nn.LSTM(INPUT_DIM, HIDDEN_DIM, N_LAYERS).to(device)
state_dict = {key.replace('a2c_network.a_rnn.rnn.', ''): value for key, value in model['model'].items() if key.startswith('a2c_network.a_rnn.rnn')}
a_rnn.load_state_dict(state_dict)

# Prepare input data
input_data = torch.zeros(1, M, INPUT_DIM, dtype=torch.float32).to(device)
random_numbers = [[random.random() for _ in range(HIDDEN_DIM * 2)] for _ in range(M)]
h = 10 * torch.tensor(random_numbers, dtype=torch.float32).reshape(1, M, HIDDEN_DIM * 2).to(device)
h.requires_grad = True

# Initialize optimizer
optimizer = torch.optim.Adam([h], lr=0.001)  # You may need to adjust learning rate based on your problem

# Other data
hx_traj = torch.zeros(MAX_ITERATIONS//SAMPLE_RATE, M, HIDDEN_DIM, dtype=torch.float32)
cx_traj = torch.zeros(MAX_ITERATIONS//SAMPLE_RATE, M, HIDDEN_DIM, dtype=torch.float32)
q_traj = torch.zeros(MAX_ITERATIONS//SAMPLE_RATE, M, dtype=torch.float32)

for epoch in range(MAX_ITERATIONS):
    # Zero out the gradients
    optimizer.zero_grad()

    a_rnn_output, (a_rnn_hx, a_rnn_cx) = a_rnn(input_data, (h[:,:,:HIDDEN_DIM].contiguous(), h[:,:,HIDDEN_DIM:].contiguous()))
    _h = torch.cat((a_rnn_hx, a_rnn_cx), dim=2)
    q = torch.norm(_h - h, dim=2)

    # Backpropagate the error
    gradient = torch.ones_like(q)
    q.backward(gradient)
    
    # Update the weights
    optimizer.step()

    if epoch % 100 == 0:
        # print only every 100 epochs
        min_index = torch.argmin(torch.norm(q, dim=0))
        print(f"epoch: {epoch}, _h min idx|: {min_index.item()}, |_h|: {torch.norm(h[:,min_index,:]).item():.2e}, q: {q[:,min_index].item():.2e}")

    if epoch % SAMPLE_RATE == 0:
        hx_traj[epoch//SAMPLE_RATE,:,:] = h[:,:,:HIDDEN_DIM].cpu()
        cx_traj[epoch//SAMPLE_RATE,:,:] = h[:,:,HIDDEN_DIM:].cpu()
        q_traj[epoch//SAMPLE_RATE,:] = q[:,:].cpu()

DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-26_09-57-04_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_08-29-41_u[-1,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_10-29-52_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# no bias but pos u and neg u, no noise/perturb
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_17-11-41_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (perturb longer)
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_08-13-39_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

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

desired_length = 200

# Extend hx_out in the first dimension
hx_out = torch.zeros((desired_length,) + hx.shape[1:], dtype=hx.dtype)
cx_out = torch.zeros((desired_length,) + cx.shape[1:], dtype=cx.dtype)

for i in range(desired_length):
    a_rnn_out, (hx, cx) = a_rnn(input, (hx.contiguous(), cx.contiguous()))
    hx_out[i,:,:] = hx
    cx_out[i,:,:] = cx




cycle_x = pd.read_csv(DATA_PATH + 'info_A_LSTM_CX_x_by_speed.csv', index_col=0)
cycle_y = pd.read_csv(DATA_PATH + 'info_A_LSTM_CX_y_by_speed.csv', index_col=0)
cycle_z = pd.read_csv(DATA_PATH + 'info_A_LSTM_CX_z1_by_speed.csv', index_col=0)

cycle_x = cycle_x.to_numpy().reshape(-1)
cycle_y = cycle_y.to_numpy().reshape(-1)
cycle_z = cycle_z.to_numpy().reshape(-1)


# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk
pca = pk.load(open(DATA_PATH + 'info_A_LSTM_CX_PCA.pkl','rb'))
scl = pk.load(open(DATA_PATH + 'A_LSTM_CX_SPEED_SCL.pkl','rb'))

cx_out1 = cx_out[-1,:,:]
cx_traj0 = cx_traj[:,3700:,:]
cx_traj1 = cx_traj[75:,3700:,:]
cx_traj2 = cx_traj[-1,3700:,:]

cx_out_pc = pca.transform(scl.transform(torch.squeeze(cx_out1).reshape(-1, 128).detach().cpu().numpy()))
cx_traj0_pc = pca.transform(scl.transform(torch.squeeze(cx_traj0).reshape(-1, 128).detach().cpu().numpy()))
cx_traj1_pc = pca.transform(scl.transform(torch.squeeze(cx_traj1).reshape(-1, 128).detach().cpu().numpy()))
cx_traj2_pc = pca.transform(scl.transform(torch.squeeze(cx_traj2).reshape(-1, 128).detach().cpu().numpy()))

import matplotlib.ticker as ticker

import numpy as np
import matplotlib.pyplot as plt

# Create a figure and subplots
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')

# Plot the second set of 3D arrays
ax1.scatter(cx_traj0_pc[:,0], cx_traj0_pc[:,1], cx_traj0_pc[:,2], c='g', s=1, alpha=0.50)

# Plot the second set of 3D arrays
ax1.scatter(cx_traj1_pc[:,0], cx_traj1_pc[:,1], cx_traj1_pc[:,2], c='g', s=10, alpha=0.75)

# Plot the second set of 3D arrays
ax1.scatter(cx_traj2_pc[:,0], cx_traj2_pc[:,1], cx_traj2_pc[:,2], c='m', s=50, alpha=1.0)

# Plot the second set of 3D arrays
ax1.scatter(cx_out_pc[:,0], cx_out_pc[:,1], cx_out_pc[:,2], c='gray', s=75)

ax1.scatter(cycle_x, cycle_y, cycle_z, c='k', s=1)

# Create a ScalarFormatter and set the desired format
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
formatter.set_powerlimits((-3, 4))  # Adjust the power limits if needed

# Apply the formatter to the axis
ax1.xaxis.set_major_formatter(formatter)
ax1.yaxis.set_major_formatter(formatter)

# Set labels and title
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_zlabel('PC 3')

# Show the plot
plt.show()


# Cluster Fixed Points

from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
import torch

def find_unique_points(tensor, eps=0.5, min_samples=5):
    # convert tensor to numpy for sklearn compatibility
    data = tensor.numpy()

    # apply DBSCAN
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(data)

    # get unique labels (clusters)
    unique_labels = np.unique(db.labels_)

    # calculate and collect the centroid for each cluster
    centroids = []
    for label in unique_labels:
        if label != -1:  # -1 label corresponds to noise in DBSCAN
            members = db.labels_ == label
            centroids.append(np.mean(data[members], axis=0))

    # convert list of centroids into a tensor
    centroids = torch.tensor(centroids)

    return centroids

fps = find_unique_points(torch.cat((hx_traj[-1,:,:], cx_traj[-1,:,:]), dim=1).detach())
fps_cx = find_unique_points(cx_traj[-1,:,:].detach())

cx_out_pc = pca.transform(scl.transform(fps_cx))


print('Finished processing.')






















