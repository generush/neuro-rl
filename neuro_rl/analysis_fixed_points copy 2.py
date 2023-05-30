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


import torch
from torch import Tensor, nn
import torch.autograd
import numpy as np

class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(CustomLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.hidden_state = None

    def reset_hidden_state(self):
        self.hidden_state = None

    def forward(self, x):
        out, self.hidden_state = self.lstm(x, self.hidden_state)
        return out

class FixedPoint(object):
    def __init__(self, fp_id: int, seq: Tensor, model: CustomLSTM=None, jacobian: Tensor=None):
        self.fp_id = fp_id
        self.seq = seq
        self.model = model
        self.eigen_values = None
        self.eigen_vectors = None
        self.jacobian = None
        self.type = None

        if jacobian is None:
            self._compute_jacobian()
        else:
            self.jacobian = jacobian

        self._analyze_stability()

    def _compute_jacobian(self):
        seq_length, input_size = self.seq.shape
        jacobian = torch.zeros(seq_length, input_size, seq_length, input_size)

        self.model.reset_hidden_state()
        output = self.model(self.seq)

        for i in range(seq_length):
            for j in range(input_size):
                output_grad = torch.zeros(seq_length, input_size).to('cuda')
                output_grad[i, j] = 1

                g = torch.autograd.grad(output, self.seq, grad_outputs=output_grad, retain_graph=True)[0]
                jacobian[i, j, :, :] = g

        self.jacobian = jacobian

    def _analyze_stability(self):
        seq_length, _ = self.seq.shape
        eigen_values = []
        eigen_vectors = []

        for i in range(seq_length):
            ev, evc = torch.linalg.eig(self.jacobian[i, :, i, :])
            eigen_values.append(ev)
            eigen_vectors.append(evc)

        self.eigen_values = torch.stack(eigen_values)
        self.eigen_vectors = torch.stack(eigen_vectors)
        self.type = classify_equilibrium(self.eigen_values)

    def to_dict(self):
        return dict(
            fp_id=self.fp_id,
            seq=self.seq.tolist(),
            eigen_values=self.eigen_values.tolist(),
            eigen_vectors=self.eigen_vectors.tolist(),
            jacobian=self.jacobian.tolist(),
            type=self.type,
        )
    
    @classmethod
    def from_dict(cls, fp_id, data_dict):
        seq = torch.Tensor(data_dict['seq']).to('cuda')
        jacobian = torch.Tensor(data_dict['jacobian']).to('cuda')
        fp = cls(fp_id, seq, jacobian=jacobian)
        return fp





model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_04-17-35-59/nn/last_AnymalTerrain_ep_2950_rew_20.14143.pth')

# no bias
model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_25-14-47-18/nn/last_AnymalTerrain_ep_2950_rew_20.2923.pth')

# no bias but pos u and neg u, no noise/perturb
model = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_27-12-07-49/nn/last_AnymalTerrain_ep_1800_rew_21.021248.pth')

DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/fixed_points.csv'


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
input = torch.zeros(1, INPUT_DIM, dtype=torch.float32).to(device)

h_data = pd.read_csv(DATA_PATH, header=0, index_col=0)
h_np = h_data.to_numpy()
h1 = h_np[0,:]
h2 = h_np[1,:]*1000
h3 = h_np[2,:]

hx = torch.Tensor(h3[:HIDDEN_DIM]).unsqueeze(dim=0).to(device).requires_grad_(True)
cx = torch.Tensor(h3[HIDDEN_DIM:]).unsqueeze(dim=0).to(device).requires_grad_(True)

n_units = cx.size(dim=1)
J_hh = torch.zeros(n_units, n_units).to(device)
J_hc = torch.zeros(n_units, n_units).to(device)
J_ch = torch.zeros(n_units, n_units).to(device)
J_cc = torch.zeros(n_units, n_units).to(device)

# Update RNN model (this updates model.hidden)
_, (hx_new, cx_new) = a_rnn(input, (hx, cx))  # LSTM returns output, (h_n, c_n)
# cx_new.requires_grad_(True)

# Compute Jacobian (change in hidden states from RNN update w.r.t change in individual hidden states)
for i in range(n_units):
    output = torch.zeros(1, n_units).to(device)
    output[:, i] = 1

    # h: RNN hidden state before update
    # h_new: RNN hidden state after update
    g_hh = torch.autograd.grad(hx_new, hx, grad_outputs=output, retain_graph=True)[0]
    g_hc = torch.autograd.grad(hx_new, cx, grad_outputs=output, retain_graph=True)[0]
    g_ch = torch.autograd.grad(cx_new, hx, grad_outputs=output, retain_graph=True)[0]
    g_cc = torch.autograd.grad(cx_new, cx, grad_outputs=output, retain_graph=True)[0]
    J_hh[i,:] = g_hh
    J_hc[i,:] = g_hc
    J_ch[i,:] = g_ch
    J_cc[i,:] = g_cc

J = result = torch.cat(
    (
        torch.cat((J_hh, J_hc), dim=1), 
        torch.cat((J_ch, J_cc), dim=1)
    ), 
    dim=0
)

J_eval, J_evec = torch.linalg.eig(J)

# validation code

import torch
from torch.autograd.functional import jacobian

# Initialize an LSTM
lstm = torch.nn.LSTM(256, 128, 1)  # Input dim is 10, output dim is 20, two layers

# Create a random input tensor
input = torch.randn(1, 1, 256)  # Sequence length is 5, batch size is 3, input dim is 10

# Initialize the hidden state
h0 = torch.randn(1, 1, 128)  # Two layers, batch size is 3, output dim is 20
c0 = torch.randn(1, 1, 128)  # Two layers, batch size is 3, output dim is 20

input = torch.zeros(1, 1, INPUT_DIM, dtype=torch.float32).to(device)

h_data = pd.read_csv(DATA_PATH, header=0, index_col=0)
h_np = h_data.to_numpy()
h1 = h_np[0,:]
h2 = h_np[1,:]
h3 = h_np[2,:]

h0 = torch.Tensor(h1[:HIDDEN_DIM]).unsqueeze(dim=0).unsqueeze(dim=0).to(device)
c0 = torch.Tensor(h1[HIDDEN_DIM:]).unsqueeze(dim=0).unsqueeze(dim=0).to(device)

# Make sure the hidden state requires gradient
h0.requires_grad_(True)
c0.requires_grad_(True)
input.requires_grad_(True)

# Concatenate h0 and c0 along a new dimension to create a single tensor
hc = torch.cat((h0.unsqueeze(0), c0.unsqueeze(0)), dim=0)

# Define a function for the LSTM with respect to input
def func_input(input):
    _, (h_out, c_out) = a_rnn(input, (h0, c0))
    return torch.cat((h_out, c_out), dim=0)

# Define a function for the LSTM with respect to hidden states
def func_hidden(hc):
    h0, c0 = torch.split(hc, 1, dim=0)
    h0 = h0.squeeze(0)
    c0 = c0.squeeze(0)
    _, (h_out, c_out) = a_rnn(input, (h0, c0))
    return torch.cat((h_out, c_out), dim=0)

# Compute the Jacobian for the output wrt the input
jacobian_matrix_input = jacobian(func_input, input)

# Compute the Jacobian for the output wrt the hidden states
jacobian_matrix_hidden = jacobian(func_hidden, hc)

# j_hc = del_h_new / del_c
J_hh = torch.squeeze(jacobian_matrix_hidden)[0,:,0,:].squeeze()
J_hc = torch.squeeze(jacobian_matrix_hidden)[0,:,1,:].squeeze()
J_ch = torch.squeeze(jacobian_matrix_hidden)[1,:,0,:].squeeze()
J_cc = torch.squeeze(jacobian_matrix_hidden)[1,:,1,:].squeeze()










# test code

import torch
from torch.autograd.functional import jacobian

# Initialize an LSTM
lstm = torch.nn.LSTM(256, 128, 1)  # Input dim is 10, output dim is 20, two layers

# Create a random input tensor
input = torch.randn(1, 1, 256)  # Sequence length is 5, batch size is 3, input dim is 10

# Initialize the hidden state
h0 = torch.randn(1, 1, 128)  # Two layers, batch size is 3, output dim is 20
c0 = torch.randn(1, 1, 128)  # Two layers, batch size is 3, output dim is 20

# Make sure the hidden state requires gradient
h0.requires_grad_(True)
c0.requires_grad_(True)
input.requires_grad_(True)

# Concatenate h0 and c0 along a new dimension to create a single tensor
hc = torch.cat((h0.unsqueeze(0), c0.unsqueeze(0)), dim=0)

# Define a function for the LSTM with respect to input
def func_input(input):
    output, _ = lstm(input, (h0, c0))
    return output

# Define a function for the LSTM with respect to hidden states
def func_hidden(hc):
    h0, c0 = torch.split(hc, 1, dim=0)
    h0 = h0.squeeze(0)
    c0 = c0.squeeze(0)
    output, (h_out, c_out) = lstm(input, (h0, c0))
    return torch.cat((h_out, c_out), dim=0)

# Compute the Jacobian for the output wrt the input
jacobian_matrix_input = jacobian(func_input, input)

# Compute the Jacobian for the output wrt the hidden states
jacobian_matrix_hidden = jacobian(func_hidden, hc)

torch.squeeze(jacobian_matrix_hidden)[0,:,0,:].squeeze()
torch.squeeze(jacobian_matrix_hidden)[0,:,1,:].squeeze()
torch.squeeze(jacobian_matrix_hidden)[1,:,0,:].squeeze()
torch.squeeze(jacobian_matrix_hidden)[1,:,1,:].squeeze()




