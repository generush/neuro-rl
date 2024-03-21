# https://plotly.com/python/3d-scatter-plots/
import os
import pickle as pk
import h5py
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from analysis.cluster import find_clusters
from analysis.jacobian import compute_jacobian, compute_jacobian_alternate, compute_jacobian_alternate2

def compute_fixed_points(model_path, processed_data_path):
    ### SET LSTM MODEL PATH

    # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_04-17-35-59/nn/last_AnymalTerrain_ep_2950_rew_20.14143.pth')

    # # no bias
    # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_25-14-47-18/nn/last_AnymalTerrain_ep_2950_rew_20.2923.pth')

    # # AnymalTerrain (1) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/o noise)
    # # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_27-12-07-49/nn/last_AnymalTerrain_ep_1800_rew_21.021248.pth')

    # # AnymalTerrain (2) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/o noise)
    # # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_29-22-48-36/nn/last_AnymalTerrain_ep_4700_rew_20.763342.pth')

    # # AnymalTerrain (3) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise)
    # # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_30-22-49-28/nn/last_AnymalTerrain_ep_4950_rew_20.344143.pth')

    # # AnymalTerrain (3a)  (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise) (earlier in training, reward = 10)
    # # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_30-22-49-28/nn/last_AnymalTerrain_ep_250_rew_10.102089.pth')

    # # AnymalTerrain (3b)  (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise) (earlier in training, reward = 15)
    # # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_30-22-49-28/nn/last_AnymalTerrain_ep_2100_rew_15.587042.pth')

    # # AnymalTerrain (3-#2) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise) (2nd model)
    # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_31-19-30-40/nn/last_AnymalTerrain_ep_7800_rew_20.086063.pth')

    # # AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM16-DIST) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=horizon_length=16) (w/o bias)
    # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_04-15-37-26/nn/last_AnymalTerrain_ep_3700_rew_20.14857.pth')

    # # AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM4-DIST500) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=4, horizon_length=16) (w/o bias)
    # # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_06-00-14-59/nn/last_AnymalTerrain_ep_6700_rew_20.21499.pth')

    # ### IMPORT TRANSFORMATION DATA

    # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-26_09-57-04_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
    # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_08-29-41_u[-1,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
    # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_10-29-52_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

    # # no bias but pos u and neg u, no noise/perturb
    # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_17-11-41_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

    # # # AnymalTerrain (perturb longer)
    # # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_08-13-39_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

    # # AnymalTerrain (1) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/o noise)
    # # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_22-30-47_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

    # # AnymalTerrain (2) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/o noise)
    # # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_13-54-22_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

    # # AnymalTerrain (3) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise)
    # # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_09-02-37_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

    # # AnymalTerrain (3a)  (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise) (earlier in training, reward = 10)
    # # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_15-54-36_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

    # # AnymalTerrain (3-#2) (perturb longer w/ noise) (with HC = (HC, CX)) (2nd model)
    # # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-01_08-11-32_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'
    # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-01_08-45-29_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

    # # AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM16-DIST500) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=horizon_length=16) (w/o bias)
    # # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-05_11-01-54_u[0.3,1.0,16]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/' # w/ noise
    # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-05_10-56-19_u[0.3,1.0,16]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/' # w/o noise

    # # AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM4-DIST500) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=4, horizon_length=16) (w/o bias)
    # # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_09-42-13_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/' # w/o noise early
    # # data_path = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_08-53-16_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/' # w/o noise

    # **LSTM16-DIST500 4/4 steps, W/ TERRAIN ()
    # model_file = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_15-24-12/nn/last_AnymalTerrain_ep_3200_rew_20.145746.pth')
    # data_path = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-08-27-16-52_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/'

    # **LSTM16-NODIST 1/4 steps (CoRL), W/ TERRAIN ()
    # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_07-03-29-04/nn/last_AnymalTerrain_ep_3800_rew_20.163399.pth')
    # data_path = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-08-27_16-52-26_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/'

    # *LSTM16-DIST500 4/4 steps, NO TERRAIN (LESS ROBUST W/O TERRAIN!!!)
    # model_file = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_14-17-13/nn/last_AnymalTerrain_ep_900_rew_20.139568.pth')
    # data_path = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-08-27_16-56-05_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/'











    # TRY RECREATING CoRL

    # **LSTM16-NODIST 1/4 steps (CoRL), W/ TERRAIN () --NOTES: CANNOT RECREATE THE 3 FIXED POINTS :( NOT SURE WHY, BUT MY RESULTS ARE INCONSISTENT, GETTING 1-3 FIXED POINTS, ALL CLOSE TO ONE ANOTHER
    # I THINK THERE IS AN ISSUE WITH THE CLUSTERING, I DON'T KNOW IF THEY ARE UNIQUE OR DUPLICATES. CHANGED # MIN POINTS FOR CLUSTER FROM 5 to 1.

    # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_07-03-29-04/nn/last_AnymalTerrain_ep_3800_rew_20.163399.pth')
    # data_path = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/data/anymal_14speed_fixedpt_orig/'

    # TESTING OUT FRONTIERS ANYMAL MODEL --NOTES: 1 STABLE FIXED POINT

    # model_file = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_15-24-12/nn/last_AnymalTerrain_ep_3200_rew_20.145746.pth')
    # data_path = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/data/anymal_14speed_fixedpt_new/'

    # TESTING OUT A1 MODEL --NOTES: 1 STABLE FIXED POINT (BUT STREAMPLOT SHOWS AS SADDLE, POSITIONED IN BETWEEN TWO ATTRACTORS).

    # model_file = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/2024-01-31-23-57_A1Terrain/nn/last_A1Terrain_ep_10100_rew_17.263123.pth')
    # data_path = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/data/a1_14speed_fixedpt_new/'

    # TESTING OUT FRONTIERS ANYMAL MODEL (NODIST-NOTERR) --NOTES: ONE STABLE FIXED POINT

    # model_file = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/2023-09-13-18-33_AnymalTerrain/nn/last_AnymalTerrain_ep_700_rew_20.361492.pth')
    # data_path = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/data/anymal_nodist_noterr_14speed_fixedpt_new/'






    # # Added old models to check their mlp, ih_l0 bias, hh_l0 bias:
    # model_file = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_04-15-37-26/nn/last_AnymalTerrain_ep_3700_rew_20.14857.pth')

    # # FRONTIERS -> ZERO BIAS hh_l0
    # model_file3=torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_15-24-12/nn/last_AnymalTerrain_ep_3200_rew_20.145746.pth')
    # model_file4=torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/2023-08-27-17-23_AnymalTerrain/nn/last_AnymalTerrain_ep_2900_rew_20.2482.pth')
    # model_file5=torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_14-17-13/nn/last_AnymalTerrain_ep_900_rew_20.139568.pth')
    # model_file6=torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/2023-09-13-18-33_AnymalTerrain/nn/last_AnymalTerrain_ep_700_rew_20.361492.pth')

    # # CORL 1/4 -> NONZERO BIAS hh_l0
    # model_file7 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_07-03-29-04/nn/last_AnymalTerrain_ep_3800_rew_20.163399.pth')

    # # CORL 4/4 -> ZERO BIAS hh_l0
    # model_file8 = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_15-24-12/nn/last_AnymalTerrain_ep_3300_rew_20.570358.pth')






    # Check if the first file exists
    print(model_path)
    model_file = torch.load(model_path)
    state_dict = {key.replace('a2c_network.a_rnn.rnn.', ''): value for key, value in model_file['model'].items() if key.startswith('a2c_network.a_rnn.rnn')}



    # # # Added old models to check their mlp, ih_l0 bias, hh_l0 bias:
    # lstm_model101 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_04-17-35-59/nn/last_AnymalTerrain_ep_2950_rew_20.14143.pth')

    # # no bias ~ BEST (no LSTM biases) AnymalTerrain_25-14-47-18
    # lstm_model102 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_25-14-47-18/nn/last_AnymalTerrain_ep_2950_rew_20.2923.pth')

    # # AnymalTerrain (1) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/o noise) (seq_len=4,horizon_length=24,pushInterval_s=15)
    # # lstm_model103 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_27-12-07-49/nn/last_AnymalTerrain_ep_1800_rew_21.021248.pth')

    # # AnymalTerrain (2) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/o noise) ~ LSTM perturb longer (no noise) AnymalTerrain_29-22-48-36
    # # lstm_model104 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_29-22-48-36/nn/last_AnymalTerrain_ep_4700_rew_20.763342.pth')

    # # AnymalTerrain (3) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise)
    # # lstm_model105 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_30-22-49-28/nn/last_AnymalTerrain_ep_4950_rew_20.344143.pth')

    # # AnymalTerrain (3a)  (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise) (earlier in training, reward = 10)
    # # lstm_model106 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_30-22-49-28/nn/last_AnymalTerrain_ep_250_rew_10.102089.pth')

    # # AnymalTerrain (3b)  (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise) (earlier in training, reward = 15)
    # # lstm_model107 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_30-22-49-28/nn/last_AnymalTerrain_ep_2100_rew_15.587042.pth')

    # # AnymalTerrain (3-#2) (pos u and neg u) (no bias) (with HC = (HC, CX)) (w/ perturb w/ noise) (2nd model)
    # lstm_model108 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_31-19-30-40/nn/last_AnymalTerrain_ep_7800_rew_20.086063.pth')

    # # AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM16-DIST) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=horizon_length=16) (w/o bias)
    # lstm_model109 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_04-15-37-26/nn/last_AnymalTerrain_ep_3700_rew_20.14857.pth')

    # # AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM4-DIST500) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=4, horizon_length=16) (w/o bias)
    # # lstm_model110 = torch.load('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_06-00-14-59/nn/last_AnymalTerrain_ep_6700_rew_20.21499.pth')
















    # get LSTM dimensions
    HIDDEN_SIZE = state_dict['weight_ih_l0'].size()[0] // 4
    INPUT_SIZE = state_dict['weight_ih_l0'].size()[1]
    N_LAYERS = max([int(key.split('_l')[-1]) for key in state_dict.keys() if key.startswith('weight_ih_l') or key.startswith('weight_hh_l')]) + 1

    # instantiate the LSTM and load weights
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    a_rnn = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS).to(device)
    a_rnn.load_state_dict(state_dict)

    # # load scaler and pca transforms
    # scl = pk.load(open('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-05_10-56-19_u[0.3,1.0,16]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/' + 'A_LSTM_HC_SPEED_SCL.pkl','rb'))
    # pca = pk.load(open('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-05_10-56-19_u[0.3,1.0,16]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/' + 'A_LSTM_HC_SPEED_PCA.pkl','rb'))


    # load scaler and pca transforms
    scl = pk.load(open(processed_data_path + 'A_LSTM_HC_SCL.pkl','rb'))
    pca = pk.load(open(processed_data_path + 'A_LSTM_HC_PCA.pkl','rb'))


    # specify parameters for fix point finder
    N_INITIAL_GUESSES = 4096 # 512
    INITIAL_GUESS_RANGE = 30
    INITIAL_GUESS_DIM = 12
    SAMPLE_RATE = 100
    MAX_ITERATIONS = 500000

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
    TOLERANCE = 4.3e-2
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


    # Convert lists to tensors
    hc_hist_fixedpt = torch.stack(hc_hist_fixedpt).squeeze()
    q_hist_fixedpt = torch.stack(q_hist_fixedpt).squeeze()

    # transform to PCA space
    hc_hist_fixedpt_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_fixedpt).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))
    hc_hist_fixedpt_ti_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_fixedpt[0,:,:]).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))
    hc_hist_fixedpt_tf_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_fixedpt[-1,:,:]).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy()))

    # Save data to hdf5
    with h5py.File(processed_data_path + 'hc_hist_fixedpt.h5', 'w') as f:
        f.create_dataset('hc_hist_fixedpt', data=hc_hist_fixedpt.detach().numpy())
    with h5py.File(processed_data_path + 'q_hist_fixedpt.h5', 'w') as f:
        f.create_dataset('q_hist_fixedpt', data=q_hist_fixedpt.detach().numpy())


    ### cluster to get unique fixed points
    fps, cnt = find_clusters(hc_hist_fixedpt[-1:,:].squeeze().detach())
    fps_pc = pca.transform(scl.transform(fps))
    pd.DataFrame(fps).to_csv(processed_data_path + 'fps.csv')
    pd.DataFrame(fps_pc).to_csv(processed_data_path + 'fps_pc.csv')
    fixed_points = torch.Tensor(fps)

    N_FIXED_POINTS = len(fps)
    input = torch.zeros(1, INPUT_SIZE).to(device)
    J_input = torch.zeros(N_FIXED_POINTS, HIDDEN_SIZE * 2, INPUT_SIZE)
    J_hidden = torch.zeros(N_FIXED_POINTS, HIDDEN_SIZE * 2, HIDDEN_SIZE * 2)
    J_hidden_pc = torch.zeros(N_FIXED_POINTS, HIDDEN_SIZE * 2, HIDDEN_SIZE * 2)
    # J_hidden2 = torch.zeros(N_FIXED_POINTS, HIDDEN_SIZE * 2, HIDDEN_SIZE * 2)
    # J_hidden3 = torch.zeros(N_FIXED_POINTS, HIDDEN_SIZE * 2, HIDDEN_SIZE * 2)
    eval = torch.zeros(N_FIXED_POINTS, HIDDEN_SIZE * 2, dtype=torch.complex64)
    evec = torch.zeros(N_FIXED_POINTS, HIDDEN_SIZE * 2, HIDDEN_SIZE * 2, dtype=torch.complex64)
    eval_pc = torch.zeros(N_FIXED_POINTS, HIDDEN_SIZE * 2, dtype=torch.complex64)
    evec_pc = torch.zeros(N_FIXED_POINTS, HIDDEN_SIZE * 2, HIDDEN_SIZE * 2, dtype=torch.complex64)

    for fp_idx, fixed_point in enumerate(fixed_points):

        J_input[fp_idx,:,:], J_hidden[fp_idx,:,:] = compute_jacobian_alternate2(a_rnn, input, fixed_point.unsqueeze(dim=0).to(device))
        # J_input2[fixed_point,:,:], J_hidden2[fixed_point,:,:] = compute_jacobian_alternate(a_rnn, input, fixed_point)
        # J_input3[fixed_point,:,:], J_hidden3[fixed_point,:,:] = compute_jacobian_alternate2(a_rnn, input, fixed_point)
        eigenvalues, eigenvectors = torch.linalg.eig(J_hidden[fp_idx, :, :])
        eval[fp_idx,:] = eigenvalues
        evec[fp_idx,:,:] = eigenvectors

        J_hidden_pc[fp_idx,:,:] = np.matmul(np.linalg.inv(pca.components_), np.matmul(J_hidden[fp_idx,:,:], pca.components_))

        # Now, you can compute the eigenvalues and eigenvectors of the PCA-transformed Jacobian.
        eigenvalues_pc, eigenvectors_pc = torch.linalg.eig(J_hidden_pc[fp_idx, :, :])
        eval_pc[fp_idx,:] = eigenvalues_pc
        evec_pc[fp_idx,:,:] = eigenvectors_pc


    # Get the real parts from the first PC of val_pc
    real_parts = eval_pc[:, 0].real

    # Find the index of the row with the smallest real part (dominant)
    min_index = torch.argmin(real_parts)

    # Get the row with the smallest real part (dominant eigenvalue)
    eval_dom_pc = eval_pc[min_index, :]
    fps_dom_pc = fps_pc[min_index, :]

    # find bounds of plots
    min_real = torch.real(eval).min()
    max_real = torch.real(eval).max()
    min_imag = torch.imag(eval).min()
    max_imag = torch.imag(eval).max()


    cycle_pc1 = pd.read_csv(processed_data_path + 'info_A_LSTM_HC_x_by_speed.csv', index_col=0)
    cycle_pc2 = pd.read_csv(processed_data_path + 'info_A_LSTM_HC_y_by_speed.csv', index_col=0)
    cycle_pc3 = pd.read_csv(processed_data_path + 'info_A_LSTM_HC_z_by_speed.csv', index_col=0)

    # df_perturb = pd.read_csv('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_17-17-18_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/RAW_DATA_AVG.csv')
    # df_perturb = pd.read_csv('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_17-57-45_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/RAW_DATA_AVG.csv')
    # df_perturb = pd.read_csv('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_18-49-34_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/RAW_DATA_AVG.csv')


    # AnymalTerrain w/ 2 LSTM (no act in obs, no zero small commands) (BEST) (2-LSTM4-DIST500) (perturb +/- 500N, 1% begin, 98% cont) (seq_len=seq_length=4, horizon_length=16) (w/o bias) DIST1000
    # df = pd.read_parquet('/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-06_17-07-25_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/RAW_DATA.parquet')



    # https://www.geeksforgeeks.org/matplotlib-pyplot-streamplot-in-python/#
    def plot_2d_streamplot(fp_idx, fixed_pts_pc, cycle_pc1, cycle_pc2, X, Y, U, V):

        # Create a figure and subplots
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # Ensure aspect ratio is equal to get a correct circle
        ax1.set_aspect('equal')


        # Plot quivers using the extracted components
        speed = np.sqrt(U**2 + V**2)
        lw = 5 * speed / speed.max()
        strm = ax1.streamplot(X, Y, U, V, density=2, linewidth=lw, color=speed/0.005, cmap ='plasma')
        plt.colorbar(strm.lines)


        # Scatter plot points for context
        # ax1.plot([EV[0] + fps_pc[1,0],fps_pc[1,0]], [EV[1] + fps_pc[1,1], fps_pc[1,1]], [EV[2] + fps_pc[1,2],fps_pc[1,2]], c='g')
        scatter2 = ax1.scatter(fixed_pts_pc[:, 0], fixed_pts_pc[:, 1], c='gray', s=50, zorder=10)
        scatter3 = ax1.scatter(fixed_pts_pc[fp_idx, 0], fixed_pts_pc[fp_idx, 1], c='k', s=60, zorder=11)


        # cycle
        ax1.plot(cycle_pc1.values[:, -1], cycle_pc2.values[:, -1], c="k", linewidth = 3.0, alpha=1)

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

        # Show the legend
        # ax1.legend()

        return fig

    def plot_2d_traj(fp_idx, fixed_pts_pc, cycle_pc1, cycle_pc2, hc_zeroinput_tf_pc):

        # Create a figure and subplots
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # Ensure aspect ratio is equal to get a correct circle
        ax1.set_aspect('equal')

        num_cols = cycle_pc1.shape[1]
        cmap = plt.cm.get_cmap('Spectral')
        colors = cmap(np.linspace(0, 1, num_cols))

        for i in range(np.shape(hc_zeroinput_tf_pc)[1]):
            # Plot the lines
            line = hc_zeroinput_tf_pc[:, i, :]
            ax1.plot(line[:, 0], line[:, 1], c='blue', alpha=0.5)

        # cycle
        ax1.plot(cycle_pc1.values[:, -1], cycle_pc2.values[:, -1], c="k", linewidth = 3.0, alpha=1)

        # Scatter plot points for context
        ax1.scatter(fixed_pts_pc[:, 0], fixed_pts_pc[:, 1], c='gray', s=50, zorder=10)
        ax1.scatter(fixed_pts_pc[fp_idx, 0], fixed_pts_pc[fp_idx, 1], c='k', s=60, zorder=11)

        # ax1.scatter(hc_perturb_pc[:, 0], hc_perturb_pc[:, 1], c='b', s=50, zorder=10)
        

        # Set labels and title
        ax1.set_xlabel('PC 1')
        ax1.set_ylabel('PC 2')
        ax1.set_title('2D Line and Quiver Plot of PC Values')

        # Show the legend
        # ax1.legend()

        return fig

    def plot_3d(fixed_pts_pc, cycle_pc1, cycle_pc2, cycle_pc3):
        
        # Create a figure and subplots
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111, projection='3d')

        # # Iterate over each line
        # for i in range(500): # hc_hist_misc_zeroinput_pc.shape[1]
        #     line = hc_hist_misc_zeroinput_pc[:, i, :]  # Get the current line
            
        #     # Plot the line
        #     ax1.plot(line[:, 0], line[:, 1], line[:, 2], c='k')

        # Scatter plot points for context
        # ax1.plot([EV[0] + fps_pc[1,0],fps_pc[1,0]], [EV[1] + fps_pc[1,1], fps_pc[1,1]], [EV[2] + fps_pc[1,2],fps_pc[1,2]], c='g')
        scatter1 = ax1.scatter(fixed_pts_pc[:, 0], fixed_pts_pc[:, 1], fixed_pts_pc[:, 2], c='k', s=50, zorder=10)

        # scatter2 = ax1.scatter(hc_perturb_pc[:, 0], hc_perturb_pc[:, 1], hc_perturb_pc[:, 2], c='k', s=1)
        scatter2 = ax1.scatter(cycle_pc1, cycle_pc2, cycle_pc3, c='r', s=1)

        # Create a ScalarFormatter and set the desired format
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        formatter.set_powerlimits((-3, 4))  # Adjust the power limits if needed

        ax1.view_init(20, 55)

        # Apply the formatter to the axis
        ax1.xaxis.set_major_formatter(formatter)
        ax1.yaxis.set_major_formatter(formatter)
        ax1.zaxis.set_major_formatter(formatter)

        # Set labels and title
        ax1.set_xlabel('PC 1')
        ax1.set_ylabel('PC 2')
        ax1.set_zlabel('PC 3')

        # Show the legend
        ax1.legend()

        return fig

    fig = plot_3d(hc_hist_fixedpt_tf_pc, cycle_pc1, cycle_pc2, cycle_pc3)

    filename = f"3d"
    fig.savefig(processed_data_path + filename + '.pdf', format='pdf', dpi=600, facecolor=fig.get_facecolor())
    fig.savefig(processed_data_path + filename + '.svg', format='svg', dpi=600, facecolor=fig.get_facecolor())

    for fp_idx, fp in enumerate(fps_pc):

        # Create the plot
        fig, ax = plt.subplots()

        # Plot the eigenvalues with lighter color and black marker outline
        ax.scatter(torch.real(eval[fp_idx,:]), torch.imag(eval[fp_idx,:]), color='lightblue', edgecolor='black', label='Eigenvalues')

        # Add a unit circle
        unit_circle = plt.Circle((0,0), 1, color='r', fill=False, label='Unit Circle')
        ax.add_artist(unit_circle)

        # Ensure aspect ratio is equal to get a correct circle
        ax.set_aspect('equal')

        # Calculate buffer for x and y limits
        buffer = 0.1

        # Setting x and y limits with buffer
        ax.set_xlim([min_real - buffer, max(1.1, max_real + buffer)])
        ax.set_ylim([min_imag - buffer, max_imag + buffer])

        plt.xlabel('Real Part')
        plt.ylabel('Imaginary Part')
        # plt.grid(True)
        plt.show()

        filename = f"fixed_point_{fp_idx}"
        fig.savefig(processed_data_path + filename + '.pdf', format='pdf', dpi=600, facecolor=fig.get_facecolor())
        fig.savefig(processed_data_path + filename + '.svg', format='svg', dpi=600, facecolor=fig.get_facecolor())

        ### STREAMPLOT
        TRAJ_TIME_LENGTH = 100
        TRAJ_XY_DENSITY = 10

        input = torch.zeros((1, TRAJ_XY_DENSITY * TRAJ_XY_DENSITY, INPUT_SIZE), device=device,  dtype=torch.float32)

        # Create the X and Y meshgrid using torch.meshgrid
        pc1_range = abs(cycle_pc1).max().max()
        pc2_range = abs(cycle_pc2).max().max()
        pc_range = max(pc1_range, pc2_range)
        X_RANGE = round(pc_range) + 5  # Adjust the value of X to set the range of the meshgrid
        x = torch.linspace(-X_RANGE, X_RANGE, TRAJ_XY_DENSITY)
        y = torch.linspace(-X_RANGE, X_RANGE, TRAJ_XY_DENSITY)
        YY, XX = torch.meshgrid(x, y)  # switch places of x and y

        # Reshape the X and Y meshgrid tensors into column vectors
        # meshgrid_tensor = torch.stack((XX.flatten(), XX.flatten()), dim=1)
        meshgrid_tensor = torch.stack((XX.flatten(), YY.flatten()), dim=1)

        # Expand the meshgrid tensor with zeros in the remaining columns
        zeros_tensor = torch.zeros(meshgrid_tensor.shape[0], 256 - 2)
        hc_zeroinput_t0_pc = torch.cat((meshgrid_tensor, zeros_tensor), dim=1).numpy()
        hc_zeroinput_t0_pc[:,2:] = fps_pc[fp_idx,2:] # PC 3+ of fixed point (SLICE THROUGH PLANE PC1-PC2)

        hc = torch.tensor(scl.inverse_transform(pca.inverse_transform(hc_zeroinput_t0_pc)), dtype=torch.float32).unsqueeze(dim=0).to(device)

        # Extend hx_out in the first dimension
        hc_hist_zeroinput = torch.zeros((TRAJ_TIME_LENGTH,) + hc.shape[1:], dtype=hc.dtype)
        hc_hist_zeroinput[0,:,:] = hc

        for i in range(TRAJ_TIME_LENGTH - 1):

            # run step
            _, (hx, cx) = a_rnn(input, (hc[:,:,:HIDDEN_SIZE].contiguous(), hc[:,:,HIDDEN_SIZE:].contiguous()))
            hc = torch.cat((hx, cx), dim=2)

            hc_hist_zeroinput[i+1,:,:] = hc

        hc_zeroinput_tf_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_zeroinput).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy())).reshape(hc_hist_zeroinput.shape)

        X = hc_zeroinput_t0_pc[:,0].reshape(TRAJ_XY_DENSITY, TRAJ_XY_DENSITY)
        Y = hc_zeroinput_t0_pc[:,1].reshape(TRAJ_XY_DENSITY, TRAJ_XY_DENSITY)
        U = (hc_zeroinput_tf_pc[1,:,0] - hc_zeroinput_tf_pc[0,:,0]).reshape(TRAJ_XY_DENSITY, TRAJ_XY_DENSITY)
        V = (hc_zeroinput_tf_pc[1,:,1] - hc_zeroinput_tf_pc[0,:,1]).reshape(TRAJ_XY_DENSITY, TRAJ_XY_DENSITY)
        fig = plot_2d_streamplot(fp_idx, fps_pc, cycle_pc1, cycle_pc2, X, Y, U, V)

        filename = f"streamplot_slice_for_fixed_point_{fp_idx}"
        fig.savefig(processed_data_path + filename + '.pdf', format='pdf', dpi=600, facecolor=fig.get_facecolor())
        fig.savefig(processed_data_path + filename + '.svg', format='svg', dpi=600, facecolor=fig.get_facecolor())

        fig = plot_2d_traj(fp_idx, fps_pc, cycle_pc1, cycle_pc2, hc_zeroinput_tf_pc)

        filename = f"2d_traj_for_fixed_point_{fp_idx}"
        fig.savefig(processed_data_path + filename + '.pdf', format='pdf', dpi=600, facecolor=fig.get_facecolor())
        fig.savefig(processed_data_path + filename + '.svg', format='svg', dpi=600, facecolor=fig.get_facecolor())

        # plot_3d_traj(hc_hist_fixedpt_tf_pc, cycle_pc1, cycle_pc2, hc_zeroinput_tf_pc)

    print('done')