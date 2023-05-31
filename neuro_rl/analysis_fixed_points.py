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
from analysis.jacobian import compute_jacobian

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


### SET LSTM MODEL PATH

lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_04-17-35-59/nn/last_AnymalTerrain_ep_2950_rew_20.14143.pth')

# no bias
lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_25-14-47-18/nn/last_AnymalTerrain_ep_2950_rew_20.2923.pth')

# no bias but pos u and neg u, no noise/perturb
lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_27-12-07-49/nn/last_AnymalTerrain_ep_1800_rew_21.021248.pth')

lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_29-22-48-36/nn/last_AnymalTerrain_ep_4700_rew_20.763342.pth')

state_dict = {key.replace('a2c_network.a_rnn.rnn.', ''): value for key, value in lstm_model['model'].items() if key.startswith('a2c_network.a_rnn.rnn')}

# get LSTM dimensions
HIDDEN_SIZE = state_dict['weight_ih_l0'].size()[0] // 4
INPUT_SIZE = state_dict['weight_ih_l0'].size()[1]
N_LAYERS = max([int(key.split('_l')[-1]) for key in state_dict.keys() if key.startswith('weight_ih_l') or key.startswith('weight_hh_l')]) + 1

# instantiate the LSTM and load weights
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
a_rnn = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS).to(device)
a_rnn.load_state_dict(state_dict)

### IMPORT TRANSFORMATION DATA

DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-26_09-57-04_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_08-29-41_u[-1,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_10-29-52_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# no bias but pos u and neg u, no noise/perturb
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_17-11-41_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (perturb longer)
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_08-13-39_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (perturb longer) (with HC = (HC, CX))
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_13-54-22_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# load scaler and pca transforms
scl = pk.load(open(DATA_PATH + 'A_LSTM_HC_SPEED_SCL.pkl','rb'))
pca = pk.load(open(DATA_PATH + 'info_A_LSTM_HC_PCA.pkl','rb'))
PCA_DIM = pca.n_components

# Set parameters for fix point finder
INITIAL_GUESS_RANGE_PC1 = 5
INITIAL_GUESS_RANGE_PC2 = 1.5
INITIAL_GUESS_RANGE_PC3 = 1.5
INITIAL_GUESS_QTY = 25
SAMPLE_RATE = 100
MAX_ITERATIONS = 10000

# initialize 
N_INITIAL_GUESSES = INITIAL_GUESS_QTY ** 3
hc0_pc = np.zeros((N_INITIAL_GUESSES, PCA_DIM), dtype=float)

# Define the range of values for each axis
x_range = np.linspace(-INITIAL_GUESS_RANGE_PC1, INITIAL_GUESS_RANGE_PC1, INITIAL_GUESS_QTY)
y_range = np.linspace(-INITIAL_GUESS_RANGE_PC2, INITIAL_GUESS_RANGE_PC2, INITIAL_GUESS_QTY)
z_range = np.linspace(-INITIAL_GUESS_RANGE_PC3, INITIAL_GUESS_RANGE_PC3, INITIAL_GUESS_QTY)

# Create a grid of coordinates using meshgrid
x, y, z = np.meshgrid(x_range, y_range, z_range)

# Stack the coordinate grids into a 3D array and reshape to 2D array
hc0_pc[:,:3] = np.stack((x, y, z), axis=-1).reshape((-1, 3))

# transform hc0_pc back to original coordinates
hc = torch.tensor(scl.inverse_transform(pca.inverse_transform(hc0_pc)), dtype=torch.float32).unsqueeze(dim=0).to(device)

# Prepare input data
input_data = torch.zeros(1, N_INITIAL_GUESSES, INPUT_SIZE, dtype=torch.float32).to(device)
# random_numbers = [[random.random() for _ in range(HIDDEN_SIZE * 2)] for _ in range(N_INITIAL_GUESSES)]
# hc = 10 * torch.tensor(random_numbers, dtype=torch.float32).reshape(1, N_INITIAL_GUESSES, HIDDEN_SIZE * 2).to(device)
hc.requires_grad = True

# Initialize optimizer
optimizer = torch.optim.Adam([hc], lr=0.0005)  # You may need to adjust learning rate based on your problem

# Other data
hc_hist_fixedpt = torch.zeros(MAX_ITERATIONS//SAMPLE_RATE, N_INITIAL_GUESSES, HIDDEN_SIZE * 2, dtype=torch.float32)
q_hist_fixedpt = torch.zeros(MAX_ITERATIONS//SAMPLE_RATE, N_INITIAL_GUESSES, dtype=torch.float32)

for epoch in range(MAX_ITERATIONS):
    # Zero out the gradients
    optimizer.zero_grad()

    _, (_h, _c) = a_rnn(input_data, (hc[:,:,:HIDDEN_SIZE].contiguous(), hc[:,:,HIDDEN_SIZE:].contiguous()))
    _hc = torch.cat((_h, _c), dim=2)
    q = torch.norm(_hc - hc, dim=2)

    # Backpropagate the error
    gradient = torch.ones_like(q)
    q.backward(gradient)
    
    # Update the weights
    optimizer.step()

    if epoch % 100 == 0:
        # print only every 100 epochs
        min_index = torch.argmin(torch.norm(q, dim=0))
        print(f"epoch: {epoch}, _hc min idx|: {min_index.item()}, |_hc|: {torch.norm(hc[:,min_index,:]).item():.2e}, q: {q[:,min_index].item():.2e}")

    if epoch % SAMPLE_RATE == 0:
        hc_hist_fixedpt[epoch//SAMPLE_RATE,:,:] = hc.cpu()
        q_hist_fixedpt[epoch//SAMPLE_RATE,:] = q[:,:].cpu()

# Save data to hdf5
with h5py.File(DATA_PATH + 'hc_hist_fixedpt.h5', 'w') as f:
    f.create_dataset('hc_hist_fixedpt', data=hc_hist_fixedpt.detach().numpy())
with h5py.File(DATA_PATH + 'q_hist_fixedpt.h5', 'w') as f:
    f.create_dataset('q_hist_fixedpt', data=q_hist_fixedpt.detach().numpy())

# PLOT EVOLUTION WITH ZERO INPUT

cycle_data = pd.read_csv(DATA_PATH + 'RAW_DATA_AVG.csv')
input = torch.zeros((1, len(cycle_data), INPUT_SIZE), device=device,  dtype=torch.float32)
hc = torch.tensor(cycle_data.loc[:,cycle_data.columns.str.contains('A_LSTM_HC')].values, device=device,  dtype=torch.float32).unsqueeze(dim=0)

desired_length = 200

# Extend hx_out in the first dimension
hc_hist_zeroinput = torch.zeros((desired_length,) + hc.shape[1:], dtype=hc.dtype)

for i in range(desired_length):
    _, (hx, cx) = a_rnn(input, (hc[:,:,:HIDDEN_SIZE].contiguous(), hc[:,:,HIDDEN_SIZE:].contiguous()))
    hc_hist_zeroinput[i,:,:] = torch.cat((hx, cx), dim=2)

cycle_pc1 = pd.read_csv(DATA_PATH + 'info_A_LSTM_CX_x_by_speed.csv', index_col=0)
cycle_pc2 = pd.read_csv(DATA_PATH + 'info_A_LSTM_CX_y_by_speed.csv', index_col=0)
cycle_pc3 = pd.read_csv(DATA_PATH + 'info_A_LSTM_CX_z1_by_speed.csv', index_col=0)

cycle_pc1 = cycle_pc1.to_numpy().reshape(-1)
cycle_pc2 = cycle_pc2.to_numpy().reshape(-1)
cycle_pc3 = cycle_pc3.to_numpy().reshape(-1)

hc_out1 = hc_hist_zeroinput[-1,:,:]
hc_hist_fixedpt_ti = hc_hist_fixedpt[0,:,:]
hc_hist_fixedpt_tnf = hc_hist_fixedpt[50:,:,:]
hc_hist_fixedpt_tf = hc_hist_fixedpt[-1,:,:]

hc_out_pc = pca.transform(scl.transform(torch.squeeze(hc_out1).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))
hc_traj_i_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_fixedpt_ti).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))
hc_traj_nf_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_fixedpt_tnf).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))
hc_traj_f_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_fixedpt_tf).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))

import matplotlib.ticker as ticker

import numpy as np
import matplotlib.pyplot as plt

# Create a figure and subplots
fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')

# Plot the second set of 3D arrays
# ax1.scatter(cx_traj0_pc[:,0], cx_traj0_pc[:,1], cx_traj0_pc[:,2], c='g', s=1, alpha=0.50)

# Plot the second set of 3D arrays
# ax1.scatter(cx_traj1_pc[:,0], cx_traj1_pc[:,1], cx_traj1_pc[:,2], c='g', s=10, alpha=0.75)

# Plot the second set of 3D arrays
# ax1.scatter(hc_traj_i_pc[:,0], hc_traj_i_pc[:,1], hc_traj_i_pc[:,2], c='k', s=50, alpha=1.0)

# Plot the second set of 3D arrays
# ax1.scatter(hc_traj_nf_pc[:,0], hc_traj_nf_pc[:,1], hc_traj_nf_pc[:,2], c='m', s=10, alpha=1.0)

# Plot the second set of 3D arrays
ax1.scatter(hc_traj_f_pc[:,0], hc_traj_f_pc[:,1], hc_traj_f_pc[:,2], c='b', s=50, alpha=1.0)

# Plot the second set of 3D arrays
# ax1.scatter(hc_out_pc[:,0], hc_out_pc[:,1], hc_out_pc[:,2], c='gray', s=75)

# ax1.scatter(cycle_pc1, cycle_pc2, cycle_pc3, c='k', s=1)

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

fps = find_clusters(hc_hist_fixedpt[-1:,:].squeeze().detach())
fps_pc = pca.transform(scl.transform(fps))

pd.DataFrame(fps).to_csv('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/fixed_points2.csv')

fixed_points = torch.Tensor(fps)



fixed_point = torch.zeros(1, HIDDEN_SIZE * 2).to(device)
input = torch.zeros(1, INPUT_SIZE).to(device)
fixed_point[0,:] = fixed_points[1,:]

J_input, J_hidden = compute_jacobian(a_rnn, input, fixed_point)
J_eval, J_evec = torch.linalg.eig(J_hidden)

torch.real(J_eval).max()



















