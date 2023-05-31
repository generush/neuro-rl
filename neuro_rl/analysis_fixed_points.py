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


### SET LSTM MODEL PATH

lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_04-17-35-59/nn/last_AnymalTerrain_ep_2950_rew_20.14143.pth')

# no bias
lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_25-14-47-18/nn/last_AnymalTerrain_ep_2950_rew_20.2923.pth')

# AnymalTerrain (1) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/o noise)
# lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_27-12-07-49/nn/last_AnymalTerrain_ep_1800_rew_21.021248.pth')

# AnymalTerrain (2) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/o noise)
# lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_29-22-48-36/nn/last_AnymalTerrain_ep_4700_rew_20.763342.pth')

# AnymalTerrain (3) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise)
# lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_30-22-49-28/nn/last_AnymalTerrain_ep_4950_rew_20.344143.pth')

# AnymalTerrain (3a)  (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise) (earlier in training, reward = 10)
lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_30-22-49-28/nn/last_AnymalTerrain_ep_250_rew_10.102089.pth')

# AnymalTerrain (3b)  (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise) (earlier in training, reward = 15)
lstm_model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_30-22-49-28/nn/last_AnymalTerrain_ep_2100_rew_15.587042.pth')


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

# # AnymalTerrain (perturb longer)
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_08-13-39_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (1) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/o noise)
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_22-30-47_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (2) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/o noise)
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_13-54-22_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (3) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise)
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_09-02-37_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (3a)  (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise) (earlier in training, reward = 10)
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_15-54-36_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'


# load scaler and pca transforms
scl = pk.load(open(DATA_PATH + 'A_LSTM_HC_SPEED_SCL.pkl','rb'))
pca = pk.load(open(DATA_PATH + 'A_LSTM_HC_SPEED_PCA.pkl','rb'))

# specify parameters for fix point finder
N_INITIAL_GUESSES = 4096
INITIAL_GUESS_RANGE = 20
INITIAL_GUESS_DIM = 12
SAMPLE_RATE = 100
MAX_ITERATIONS = 100000

# Generate random numbers within the specified bounds
random_numbers = [[random.uniform(-1, 1) for _ in range(INITIAL_GUESS_DIM)] for _ in range(N_INITIAL_GUESSES)]
hc0_pc = torch.zeros((N_INITIAL_GUESSES, HIDDEN_SIZE * 2), dtype=torch.float32)
hc0_pc[:,:INITIAL_GUESS_DIM] = INITIAL_GUESS_RANGE * torch.tensor(random_numbers, dtype=torch.float32)
hc0 = torch.tensor(scl.inverse_transform(pca.inverse_transform(hc0_pc)), dtype=torch.float32).unsqueeze(dim=0).to(device)

# prepare input data
input_data = torch.zeros(1, N_INITIAL_GUESSES, INPUT_SIZE, dtype=torch.float32).to(device)

# set hc to initial guesses
hc = hc0
hc.requires_grad = True

# specify parameters for optimizer
LEARNING_RATE = 0.001
TOLERANCE = 5e-2
optimizer = torch.optim.Adam([hc], lr=LEARNING_RATE)  # You may need to adjust learning rate based on your problem

# initialize as empty lists
hc_hist_fixedpt = [] 
q_hist_fixedpt = []
delta_hc = torch.inf
q = torch.empty((1, N_INITIAL_GUESSES)).fill_(float('inf'))

for epoch in range(MAX_ITERATIONS):

    # zero out the gradients
    optimizer.zero_grad()
    
    # append hc (hidden state) and q (velocity) to history
    if epoch % SAMPLE_RATE == 0:
        hc_hist_fixedpt.append(hc.cpu())  # Append to the list
        q_hist_fixedpt.append(q[:,:].cpu())

        # compute change in hidden state (for exit criteria)
        if len(hc_hist_fixedpt) >= 2:
            delta_hc = (hc_hist_fixedpt[-1]-hc_hist_fixedpt[-2]).norm()

        # printing
        max_index = torch.argmax(torch.norm(q, dim=0))
        max_elem = torch.max(q).item()
        print(f"\
            epoch: {epoch}, _hc min idx|: {max_index.item()}, \
            |_hc|: {torch.norm(hc[:,max_index,:]).item():.2e}, \
            q: {q[:,max_index].item():.2e}, \
            delta_hc: {delta_hc:.2e}"
        )

        # exit if found fixed points
        if  delta_hc < TOLERANCE:  # Stopping criterion
            print(f"Stopping criterion reached at epoch: {epoch}")
            break
    
    # run optimization step
    _, (_h, _c) = a_rnn(input_data, (hc[:,:,:HIDDEN_SIZE].contiguous(), hc[:,:,HIDDEN_SIZE:].contiguous()))
    _hc = torch.cat((_h, _c), dim=2)
    q = torch.norm(_hc - hc, dim=2)

    # backpropagate the error
    gradient = torch.ones_like(q)
    q.backward(gradient)
    
    # Update the weights
    optimizer.step()




### cluster to get unique fixed points
fps, cnt = find_clusters(hc_hist_fixedpt[-1:,:].squeeze().detach())
fps_pc = pca.transform(scl.transform(fps))
pd.DataFrame(fps).to_csv('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/fixed_points2.csv')
fixed_points = torch.Tensor(fps)

### compute jacobian and 
fixed_point = torch.zeros(1, HIDDEN_SIZE * 2).to(device)
input = torch.zeros(1, INPUT_SIZE).to(device)
fixed_point[0,:] = fixed_points[0,:]

J_input, J_hidden = compute_jacobian_alternate(a_rnn, input, fixed_point)
J_input2, J_hidden2 = compute_jacobian_alternate(a_rnn, input, fixed_point)
J_input3, J_hidden3 = compute_jacobian_alternate2(a_rnn, input, fixed_point)
J_eval, J_evec = torch.linalg.eig(J_hidden)
torch.real(J_eval).max()





# # Convert lists to tensors
# hc_hist_fixedpt = torch.stack(hc_hist_fixedpt).squeeze()
# q_hist_fixedpt = torch.stack(q_hist_fixedpt).squeeze()

# # Save data to hdf5
# with h5py.File(DATA_PATH + 'hc_hist_fixedpt.h5', 'w') as f:
#     f.create_dataset('hc_hist_fixedpt', data=hc_hist_fixedpt.detach().numpy())
# with h5py.File(DATA_PATH + 'q_hist_fixedpt.h5', 'w') as f:
#     f.create_dataset('q_hist_fixedpt', data=q_hist_fixedpt.detach().numpy())

# # PLOT EVOLUTION WITH ZERO INPUT

# cycle_data = pd.read_csv(DATA_PATH + 'RAW_DATA_AVG.csv')
# input = torch.zeros((1, len(cycle_data), INPUT_SIZE), device=device,  dtype=torch.float32)
# hc0 = torch.tensor(cycle_data.loc[:,cycle_data.columns.str.contains('A_LSTM_HC')].values, device=device,  dtype=torch.float32).unsqueeze(dim=0)
# hc = hc0

# desired_length = 1000

# # Extend hx_out in the first dimension
# hc_hist_zeroinput = torch.zeros((desired_length,) + hc.shape[1:], dtype=hc.dtype)

# for i in range(desired_length):

#     # add hidden state to history
#     hc_hist_zeroinput[i,:,:] = hc

#     # run step
#     _, (hx, cx) = a_rnn(input, (hc[:,:,:HIDDEN_SIZE].contiguous(), hc[:,:,HIDDEN_SIZE:].contiguous()))
#     hc = torch.cat((hx, cx), dim=2)

# cycle_pc1 = pd.read_csv(DATA_PATH + 'info_A_LSTM_HC_x_by_speed.csv', index_col=0)
# cycle_pc2 = pd.read_csv(DATA_PATH + 'info_A_LSTM_HC_y_by_speed.csv', index_col=0)
# cycle_pc3 = pd.read_csv(DATA_PATH + 'info_A_LSTM_HC_z1_by_speed.csv', index_col=0)

# cycle_pc1 = cycle_pc1.to_numpy().reshape(-1)
# cycle_pc2 = cycle_pc2.to_numpy().reshape(-1)
# cycle_pc3 = cycle_pc3.to_numpy().reshape(-1)

# hc_hist_zeroinput_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_zeroinput).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))
# hc_hist_zeroinput_ti_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_zeroinput[0,:,:]).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))
# hc_hist_zeroinput_tf_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_zeroinput[-1,:,:]).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))

# hc_hist_fixedpt_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_fixedpt).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))
# hc_hist_fixedpt_ti_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_fixedpt[0,:,:]).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))
# hc_hist_fixedpt_tf_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_fixedpt[-1,:,:]).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))

# import matplotlib.ticker as ticker

# import numpy as np
# import matplotlib.pyplot as plt

# # Create a figure and subplots
# fig = plt.figure()
# ax1 = fig.add_subplot(121, projection='3d')

# # zero input: final points
# ax1.scatter(hc_hist_zeroinput_tf_pc[:,0], hc_hist_zeroinput_tf_pc[:,1], hc_hist_zeroinput_tf_pc[:,2], c='g', s=100, alpha=1.0)

# # zero input: initial points
# # ax1.scatter(hc_hist_zeroinput_ti_pc[:,0], hc_hist_zeroinput_ti_pc[:,1], hc_hist_zeroinput_ti_pc[:,2], c='k', s=20, alpha=1.0)

# # zero input: history
# ax1.scatter(hc_hist_zeroinput_pc[:,0], hc_hist_zeroinput_pc[:,1], hc_hist_zeroinput_pc[:,2], c='gray', s=0.1, alpha=0.5)

# # fixed points
# ax1.scatter(hc_hist_fixedpt_tf_pc[:,0], hc_hist_fixedpt_tf_pc[:,1], hc_hist_fixedpt_tf_pc[:,2], c='b', s=50, alpha=1.0)

# # fixed point: initial guesses
# # ax1.scatter(hc_hist_fixedpt_ti_pc[:,0], hc_hist_fixedpt_ti_pc[:,1], hc_hist_fixedpt_ti_pc[:,2], c='k', s=1, alpha=1.0)

# # fixed point: history of guesses
# # ax1.scatter(hc_hist_fixedpt_pc[:,0], hc_hist_fixedpt_pc[:,1], hc_hist_fixedpt_pc[:,2], c='m', s=10, alpha=0.5)


# # Plot the second set of 3D arrays
# # ax1.scatter(hc_out_pc[:,0], hc_out_pc[:,1], hc_out_pc[:,2], c='gray', s=75)

# ax1.scatter(cycle_pc1, cycle_pc2, cycle_pc3, c='r', s=1)

# # Create a ScalarFormatter and set the desired format
# formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_scientific(False)
# formatter.set_powerlimits((-3, 4))  # Adjust the power limits if needed

# # Apply the formatter to the axis
# ax1.xaxis.set_major_formatter(formatter)
# ax1.yaxis.set_major_formatter(formatter)

# # Set labels and title
# ax1.set_xlabel('PC 1')
# ax1.set_ylabel('PC 2')
# ax1.set_zlabel('PC 3')

# # Show the plot
# plt.show()














### Not sure if this is correct...

# For the scaler, we need to consider the square root of the variance 
# as the standard deviation is used for scaling
scaled_variance = np.sqrt(scl.var_ + 1e-10)  # adding a small value to avoid division by zero

# Reshape the variance to match matrix multiplication requirements
scaled_variance = scaled_variance.reshape((1, len(scaled_variance)))

J_scaled_input = J_input / scaled_variance.T
J_scaled_hidden = J_hidden / scaled_variance.T

J_pca_input = np.matmul(pca.components_, J_scaled_input)
J_pca_hidden = np.matmul(pca.components_, J_scaled_hidden)

# Now, you can compute the eigenvalues and eigenvectors of the PCA-transformed Jacobian.
eigvals_input, eigvecs_input = torch.linalg.eig(J_pca_input)
eigvals_hidden, eigvecs_hidden = torch.linalg.eig(J_pca_hidden)

# These are the eigenvalues and eigenvectors of the system in PCA space.







































input = torch.zeros((1, 100, INPUT_SIZE), device=device,  dtype=torch.float32)
hc0 = 1 * torch.tensor(scl.inverse_transform(pca.inverse_transform(hc0_pc[:100,:])), dtype=torch.float32).unsqueeze(dim=0).to(device)
hc = torch.zeros((1, 100, HIDDEN_SIZE * 2), device=device,  dtype=torch.float32)
hc[0,:,:] = hc0 + torch.tensor(fps[0,:]).to(device)

desired_length = 100

# Extend hx_out in the first dimension
hc_hist_misc_zeroinput = torch.zeros((desired_length,) + hc.shape[1:], dtype=hc.dtype)

for i in range(desired_length):

    # add hidden state to history
    hc_hist_misc_zeroinput[i,:,:] = hc

    # run step
    _, (hx, cx) = a_rnn(input, (hc[:,:,:HIDDEN_SIZE].contiguous(), hc[:,:,HIDDEN_SIZE:].contiguous()))
    hc = torch.cat((hx, cx), dim=2)

hc_hist_misc_zeroinput_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_misc_zeroinput).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy())).reshape(hc_hist_misc_zeroinput.shape)

import matplotlib.ticker as ticker
import numpy as np
import matplotlib.pyplot as plt

# Create a figure and subplots
fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

# Iterate over each line
for i in range(100): # hc_hist_misc_zeroinput_pc.shape[1]
    line = hc_hist_misc_zeroinput_pc[:, i, :]  # Get the current line
    
    # Plot the line
    ax1.plot(line[:, 0], line[:, 1], line[:, 2], c='k')

# Scatter plot points for context
# ax1.scatter(hc_hist_nearsaddle_zeroinput_pc[:, 0], hc_hist_nearsaddle_zeroinput_pc[:, 1], hc_hist_nearsaddle_zeroinput_pc[:, 2], c='k', s=1, alpha=1.0, label='zero input: final points')
# ax1.scatter(cycle_pc1, cycle_pc2, cycle_pc3, c='r', s=1, label='cycle')
ax1.scatter(hc_hist_fixedpt_tf_pc[:, 0], hc_hist_fixedpt_tf_pc[:, 1], hc_hist_fixedpt_tf_pc[:, 2], c='b', s=50, alpha=1.0, label='fixed points')

# Create a ScalarFormatter and set the desired format
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(False)
formatter.set_powerlimits((-3, 4))  # Adjust the power limits if needed

# Apply the formatter to the axis
ax1.xaxis.set_major_formatter(formatter)
ax1.yaxis.set_major_formatter(formatter)
ax1.zaxis.set_major_formatter(formatter)

# Set labels and title
ax1.set_xlabel('PC 1')
ax1.set_ylabel('PC 2')
ax1.set_zlabel('PC 3')
ax1.set_title('3D Line Plot of PC Values')

# Show the legend
ax1.legend()

# Show the plot
plt.show()


