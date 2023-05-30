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
MAX_ITERATIONS = 20000

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
h = torch.nn.Parameter(torch.cat((h1, h2), dim=1))  # Change h into an nn.Parameter
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

cx_out1 = cx_out[:,:,:]
cx_traj1 = cx_traj[75:,3700:,:]

cx_out_pc = pca.transform(scl.transform(torch.squeeze(cx_out1).reshape(-1, 128).detach().cpu().numpy()))
cx_traj_pc = pca.transform(scl.transform(torch.squeeze(cx_traj1).reshape(-1, 128).detach().cpu().numpy()))

import matplotlib.ticker as ticker

import numpy as np
import matplotlib.pyplot as plt

# Create a figure and subplots
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')

# Plot the second set of 3D arrays
ax1.scatter(cx_traj_pc[:,0], cx_traj_pc[:,1], cx_traj_pc[:,2], c='b', s=1)

# Plot the second set of 3D arrays
ax1.scatter(cx_out_pc[:,0], cx_out_pc[:,1], cx_out_pc[:,2], c='m', s=1)

ax1.scatter(cycle_x, cycle_y, cycle_z, c='k', s=1)

# Create a ScalarFormatter and set the desired format
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
formatter.set_powerlimits((-3, 4))  # Adjust the power limits if needed

# Apply the formatter to the axis
ax1.xaxis.set_major_formatter(formatter)
ax1.yaxis.set_major_formatter(formatter)

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

# Stability Test

# for i in range(3):
#     inp = input_data[0,0,:].unsqueeze(0) # Add batch dimension
#     h = fps[0,:128].unsqueeze(0).to(device) # Add batch dimension
#     c = fps[0,128:].unsqueeze(0).to(device)  # Add batch dimension
    
#     h_full = torch.cat((h,c), dim=1).requires_grad_() 
#     outp, (h_out, c_out) = a_rnn(inp, (h, c))
#     h_full_out = torch.cat((h_out,c_out), dim=1).requires_grad_() 

#     """Computes jacobian at the hidden state fixed point of the RNN."""
#     n_units = h.size(dim=1)
#     jacobian = torch.zeros(n_units, n_units)

#     # Compute Jacobian (change in hidden states from RNN update w.r.t change in individual hidden states)
#     for i in range(n_units):
#         output = torch.zeros((1, n_units)).to('cuda')
#         output[:,i] = 1

#         # self.h: RNN hidden state before update
#         # self.model.hx: RNN hidden state after update
#         g = torch.autograd.grad(c, c_out, grad_outputs=output, retain_graph=True)[0]
#         jacobian[i,:] = g
    
#     self.jacobian = jacobian







class RNNJacobian:
    def __init__(self, a_rnn, h, constant_input):
        self.a_rnn = a_rnn
        self.h = h
        self.hx = h
        self.constant_input = constant_input
        self.jacobian = None
    
    def update():
        self.hx = self.a_rnn(self.constant_input, self.h)

    def compute_jacobian(self):
        """Computes Jacobian at the hidden state fixed point of the RNN."""
        n_units = self.h.size(dim=0)
        jacobian = torch.zeros(n_units, n_units)

        # Initialize hidden state
        # self.a_rnn.hx = self.h.requires_grad_(True)
        # self.h.requires_grad_(True)

        # Update RNN model (this updates self.a_rnn.hx)
        self.update()

        # Compute Jacobian (change in hidden states from RNN update w.r.t change in individual hidden states)
        for i in range(n_units):
            output = torch.zeros(1, n_units).to('cuda')
            output[0, i] = 1

            # self.h: RNN hidden state before update
            # self.a_rnn.hx: RNN hidden state after update
            grad_output = output.expand(self.a_rnn.hx.size())  # Expand the shape of grad_output to match output
            g = torch.autograd.grad(self.hx, self.h, grad_outputs=grad_output, retain_graph=True)[0]
            jacobian[i,:] = g
        
        self.jacobian = jacobian

constant_input = torch.zeros((1,1,INPUT_DIM)).to(device)  # Replace ... with appropriate shape and values for constant input
h = torch.unsqueeze(fps[0,:],dim=0).to(device)
jacobian_calculator = RNNJacobian(a_rnn, h, constant_input)

# Compute Jacobian
jacobian_calculator.compute_jacobian()


print('Finished processing.')






















