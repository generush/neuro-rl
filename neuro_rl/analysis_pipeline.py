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
from analysis.analyze_traj import analyze_avg_traj
from analysis.analyze_pca import analyze_pca
from analysis.analyze_pca_speed_axis import analyze_pca_speed_axis
from analysis.analyze_tangling import analyze_tangling
from plotting.dashboard import run_dashboard

import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

import time

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

# no bias
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-26_09-57-04_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'

# no bias but pos u and neg u
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_08-29-41_u[-1,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_10-29-52_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# no bias but pos u and neg u, no noise/perturb
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-27_17-11-41_u[-1,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# fixed CX1, CX2
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-28_20-55-55_u[1,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/'

# LSTM trained w/ perturb, tested w/ perturb N=1
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_08-06-39_u[1,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/'

# LSTM trained w/o perturb, tested w/ perturb N=1
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_08-08-00_u[1,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/'

# LSTM trained w/ perturb, tested w/o perturb N=1
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_08-12-29_u[1,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/'

# LSTM trained w/o perturb, tested w/o perturb N=1
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_08-13-18_u[1,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/'

# LSTM trained w/o perturb, tested w/o perturb N=700 --> VERIFIED IT LOOKS NORMAL
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_08-28-38_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'

# LSTM trained w/ perturb, tested w/o perturb N=700
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_08-32-16_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'

# FF trained w/ perturb, tested w/o perturb N=700
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_09-42-49_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_09-49-43_u[0.4,1.0,7]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'

# ShadowHand 4 rolls
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_11-28-20_u[-90.0,90.0,4]_v[0.0,0.0,0]_r[0.0,0.0,0]/'

# ShadowHand 4 same block sizes
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_12-15-43_u[90.0,90.0,4]_v[0.0,0.0,0]_r[0.0,0.0,0]/'

# ShadowHand 4 diff block sizes 0.95-1.05
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_12-31-34_u[90.0,90.0,4]_v[0.0,0.0,0]_r[0.0,0.0,0]/'

# ShadowHand 4 diff block sizes 0.8-1.2
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_12-34-22_u[90.0,90.0,4]_v[0.0,0.0,0]_r[0.0,0.0,0]/'

# ShadowHand 4 diff block masses 0.5-2.0
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-29_12-38-14_u[90.0,90.0,4]_v[0.0,0.0,0]_r[0.0,0.0,0]/'

# AnymalTerrain (perturb longer)
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_08-13-39_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (1) (no bias) no bias but pos u and neg u, no noise/perturb (with HC = (HC, CX))
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_22-30-47_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (2) (perturb longer) (with HC = (HC, CX))
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-30_13-54-22_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (3) (perturb longer w/ noise) (with HC = (HC, CX))
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_09-02-37_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (3a) (perturb longer w/ noise) (with HC = (HC, CX)) (earlier in training, reward = 10)
# DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_15-54-36_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

# AnymalTerrain (3-dist) (perturb longer w/ noise) (with HC = (HC, CX)) (perturb during experiment)
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_17-17-18_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[100]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_17-57-45_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_18-49-34_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/' # 150N in +v direction
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-31_19-05-00_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/' # 1200N in +v direction 

# AnymalTerrain (3-#2) (perturb longer w/ noise) (with HC = (HC, CX)) (2nd model)
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-01_08-11-32_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/' # w/o noise
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-01_08-45-29_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/' # w/ noise
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-06-02_10-25-10_u[-1.0,1.0,21]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/' # w/ noise





DATASETS = [
    # 'OBS',
    # 'ACT',
    # 'A_MLP_XX',
    'A_LSTM_HC',
    # 'A_LSTM_CX',
    # 'A_LSTM_HX',
    # 'A_LSTM_C1X',
    # 'A_LSTM_C2X',
    # 'C_MLP_XX',
    # 'C_LSTM_CX',
    # 'C_LSTM_HX',
    # 'A_GRU_HX',
    # 'C_GRU_HX',
]

AVG = True

start = time.process_time()

if AVG:

    df_avg = analyze_avg_traj(DATA_PATH)
    print('Finished analyze_traj', time.process_time() - start)

    # df_avg = analyze_cycle(DATA_PATH)
    # print('Finished analyze_cycle', time.process_time() - start)

    data_w_tangling = analyze_tangling(DATA_PATH, DATASETS, '_AVG')
    print('Finished analyze_tangling', time.process_time() - start)

    # analyze_pca_speed_axis(DATA_PATH, DATASETS, '_AVG_WITH_TANGLING')
    # print('Finished analyze_pca', time.process_time() - start)

    analyze_pca(DATA_PATH, DATASETS)
    print('Finished analyze_pca', time.process_time() - start)

    run_dashboard(DATA_PATH)

else:
    analyze_pca(DATA_PATH, DATASETS)
    print('Finished analyze_pca', time.process_time() - start)

    run_dashboard(DATA_PATH)
