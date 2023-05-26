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

# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-15_21-25-03_u[-1,-0.4,7]_v[0]_r[0]_n[100]/'
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-15_21-20-32_u[0.4,1,7]_v[0]_r[0]_n[100]/'
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-15_21-33-03_u[0]_v[-1,-0.4,7]_r[0]_n[100]/'
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-15_21-36-23_u[0]_v[0.4,1,7]_r[0]_n[100]/'
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-15_21-39-15_u[1]_v[0]_r[-1,1,7]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-15_22-37-25_u[1]_v[0]_r[-1,1,2]_n[100]/'
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-15_22-36-09_u[0]_v[-1,1,2]_r[0]_n[100]/'
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-15_22-34-50_u[-1,1,2]_v[0]_r[0]_n[100]/'


# exp 1
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/exp1_2023-05-16_23-21-39_u[1]_v[0]_r[0]_n[100]_w_noise/'

# exp 1a (perturbation)
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/exp1a_2023-05-16_02-29-51_u[1]_v[0]_r[0]_n[1]__wo_noise_w_velperturb/'

# exp 2
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/exp2a_2023-05-17_00-01-33_u[-1,1,2]_v[0]_r[0]_n[100]_w_noise/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/exp2b_2023-05-17_00-09-06_u[0]_v[-1,1,2]_r[0]_n[100]_w_noise/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/exp2c_2023-05-17_00-13-34_u[1]_v[0]_r[-1,1,2]_n[100]_w_noise/'

# exp3
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/exp3_2023-05-17_00-53-30_u[0.4,1.0,7]_v[0]_r[0]_n[100]_w_noise/'
# # need to generate the rest of the results for the Suppl Mat

# # actually less tangling?
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/exp_extra4_2023-05-17_01-25-18_u[0.4,1,7]_v[0]_r[0]_n[100]_w_noise_w_act_in_obs/'

# # no major difference
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-17_01-38-44_u[0.4,1,7]_v[0]_r[0]_n[100]_w_noise_trainedwofriction/'
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-17_01-48-03_u[0.4,1,7]_v[0]_r[0]_n[100]_w_noise_trainedwofrictionterrain/'

# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-17_02-01-56_u[0.4,1.0,7]_v[0]_r[0]_n[100]_w_noise/'

# # grabbign PC1 vector, PC2 vector, and speed axis vector for a lstm cx, and % variance explained ratio vector for act and a lstm cx
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-17_02-18-59_u[-1,1.0,2]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-17_02-45-04_u[0,0.0,1]_v[-1.0,1.0,2]_r[0.0,0.0,1]_n[100]/'
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-17_02-50-13_u[1.0,1.0,1]_v[0.0,0.0,1]_r[-1.0,1.0,2]_n[100]/'

# want to see smearing
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-17_11-53-03_u[0.0,0.0,1]_v[0.4,1.0,7]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-17_11-58-24_u[-1.0,-0.4,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'




DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/supp/exp3_modspeeds/2023-05-23_08-40-40_u[-1.0,-0.4,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/supp/exp3_modspeeds/2023-05-23_08-47-27_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/supp/exp3_modspeeds/2023-05-23_09-04-55_u[0.0,0.0,1]_v[-1.0,-0.4,7]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/supp/exp3_modspeeds/2023-05-23_09-11-17_u[0.0,0.0,1]_v[0.4,1.0,7]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/supp/exp3_modspeeds/2023-05-23_09-15-55_u[1.0,1.0,1]_v[0.0,0.0,1]_r[-1.0,-0.4,7]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/supp/exp3_modspeeds/2023-05-23_09-22-09_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.4,1.0,7]_n[100]/'

# supp: more models
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/supp/exp3b_modspeeds_moremodels/2023-05-23_09-28-51_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'


# supp: exp1_onespeed_obs_hx
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/supp/exp1_onespeed_obs_hx/2023-05-23_10-14-33_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'



# supp: exp1_onespeed_obs_hx w/o act!!
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-23_11-39-51_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'

DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-23_11-52-12_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'

# exp1_onespeed (re-analyze)
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/exp1_onespeed/exp1_2023-05-16_23-21-39_u[1]_v[0]_r[0]_n[100]_w_noise (copy)/'


DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-24_10-43-18_u[-1.0,1.0,2]_v[-1.0,1.0,2]_r[-1.0,1.0,2]_n[100]/'

# debug tangling
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/exp3_modspeeds/exp3_2023-05-17_00-53-30_u[0.4,1.0,7]_v[0]_r[0]_n[100]_w_noise/'


model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_04-17-35-59/nn/last_AnymalTerrain_ep_2950_rew_20.14143.pth')

# no bias
model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_25-14-47-18/nn/last_AnymalTerrain_ep_2950_rew_20.2923.pth')


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import random

from torch.autograd import grad


# Create an instance of the A2C network
input_size = 256  # Replace with the appropriate input size
hidden_size = 128  # Replace with the desired hidden size of the LSTM
num_layers = 1  # Replace with the desired number of LSTM layers
output_size = 128  # Replace with the appropriate output size

a_rnn = nn.LSTM(input_size, hidden_size, num_layers)

# Load the LSTM model weights
state_dict = {
    key.replace('a2c_network.a_rnn.rnn.', ''): value
    for key, value in model['model'].items()
    if key.startswith('a2c_network.a_rnn.rnn')
}
a_rnn.load_state_dict(state_dict)

# Set the device for the LSTM
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
a_rnn.to(device)

# Perform forward pass

# random_numbers = [random.random() for _ in range(hidden_size * 2)]
# h = torch.tensor(random_numbers, dtype=torch.float32).reshape(1, hidden_size * 2).to(device)

M = 4096

# cx0 = torch.zeros(1, hidden_size, dtype=torch.float32).to(device)


max_iters = 5000

gamma = 0.1 * torch.ones(1, M, 1, dtype=torch.float32).to(device)

input_data = torch.zeros(1, M, input_size, dtype=torch.float32).to(device)

random_numbers = [[random.random() for _ in range(hidden_size * 2)] for _ in range(M)]
h = 5 * torch.tensor(random_numbers, dtype=torch.float32).reshape(1, M, hidden_size * 2).to(device) # 5 --> _h=5.72 # 1 --> _h=0 # 0.01 --> _h=0

# h2 = 5 * torch.tensor(random_numbers, dtype=torch.float32).reshape(1, M//2, hidden_size * 2).to(device) # 5 --> _h=5.72 # 1 --> _h=0 # 0.01 --> _h=0
# h = torch.cat((h1, h2),dim=1)

h.requires_grad = True
h.retain_grad()

hx_traj = torch.zeros(max_iters//100, M, hidden_size, dtype=torch.float32)
cx_traj = torch.zeros(max_iters//100, M, hidden_size, dtype=torch.float32)
q_traj = torch.zeros(max_iters//100, M, dtype=torch.float32)


q = torch.zeros(1, M, dtype=torch.float32).to(device)
q_last = torch.zeros(1, M, dtype=torch.float32).to(device)
q_last_last = torch.zeros(1, M, dtype=torch.float32).to(device)

for epoch in range(max_iters):
    a_rnn_output, (a_rnn_hx, a_rnn_cx) = a_rnn(input_data, (h[:,:,:128].contiguous(), h[:,:,128:].contiguous()))
    _h = torch.cat((a_rnn_hx, a_rnn_cx), dim=2)

    # Compute speed
    q = torch.norm(h - _h, dim=2)
    # print(epoch, gamma, q, abs(q - q_last), _h - h)

    # print stats
    norm_q = torch.norm(q, dim=0)
    min_value, min_index = norm_q.min(dim=0)

    print(
        f"epoch: {epoch}"
        f"\t _h min idx|: {min_index.item()}"
        f"\t |_h|: {torch.norm(h[:,min_index,:]).item():.2e}"
        f"\t q: {q[:,min_index].item():.2e}"
        f"\t |q - q_last|: {torch.norm(q[:,min_index] - q_last[:,min_index]).item():.2e}"
        f"\t gamma: {gamma[:, min_index, :].item():.2e}"
        f"\t |q - q_last_last|: {torch.norm(q[:,min_index] - q_last_last[:,min_index]).item():.2e}"
    )

    # print whole tensor _h
    # norms = torch.norm(_h, dim=2)
    # norms_str = ", ".join(f"{norm:.3f}" for norm in norms.flatten())
    # print(f"\t |_h|: {norms_str}")

    # Step in the direction of decreasing speed

    # Now, let's create a tensor of the same shape as `output`
    gradient = torch.ones_like(q)

    # Now you can compute the gradient
    q.backward(gradient)

    # Update hidden state, h.grad = dq/dh
    h = h - gamma * h.grad
    h.retain_grad()

    # Reduce gamma if q bounces back and forth between two values
    mask = torch.norm(q - q_last, dim=0) > 10 * torch.norm(q - q_last_last, dim=0)
    mask = mask.unsqueeze(dim=1).unsqueeze(dim=0)
    
    gamma *= torch.where(mask, 0.99, 1.0)

    # if torch.norm(q[0,0] - q_last[0,0]) > 10 * torch.norm(q[0,0] - q_last_last[0,0]): # TO DO: FIX POINTS BOUNCING BACK AND FORTH AND NOT CONVERGING
    #     gamma *= 0.99
        # random_numbers = [[random.random() for _ in range(hidden_size * 2)] for _ in range(M)]
        # h += 0.001 * gamma * torch.tensor(random_numbers, dtype=torch.float32).reshape(1, M, hidden_size * 2).to(device)
        # random_numbers =  [random.random() for _ in range(hidden_size * 2)]
        # h += 0.001 * gamma * torch.tensor(random_numbers, dtype=torch.float32).reshape(1, hidden_size * 2).to(device)

    # Update previous speeds
    q_last_last = q_last
    q_last = q

    if epoch % 100 == 0:
        hx_traj[epoch//100,:,:] = _h[:,:,:128].detach().cpu()
        cx_traj[epoch//100,:,:] = _h[:,:,128:].detach().cpu()
        q_traj[epoch//100,:] = q[:,:].detach().cpu()

DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-26_09-57-04_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'

import h5py

# Create an HDF5 file
with h5py.File(DATA_PATH + 'cx_traj.h5', 'w') as f:
    # Create a dataset in the file and write the tensor data
    f.create_dataset('cx_traj', data=cx_traj.numpy())

# Create an HDF5 file
with h5py.File(DATA_PATH + 'hx_traj.h5', 'w') as f:
    # Create a dataset in the file and write the tensor data
    f.create_dataset('hx_traj', data=hx_traj.numpy())

# Create an HDF5 file
with h5py.File(DATA_PATH + 'q_traj.h5', 'w') as f:
    # Create a dataset in the file and write the tensor data
    f.create_dataset('q_traj', data=q_traj.numpy())