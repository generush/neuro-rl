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


DATASETS = [
    'OBS',
    'ACT',
    # 'A_MLP_XX',
    'A_LSTM_CX',
    'A_LSTM_HX',
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
    df_avg = analyze_cycle(DATA_PATH)
    print('Finished analyze_cycle', time.process_time() - start)

    data_w_tangling = analyze_tangling(DATA_PATH, DATASETS, '_AVG')
    print('Finished analyze_tangling', time.process_time() - start)

    analyze_pca_speed_axis(DATA_PATH, DATASETS, '_AVG_WITH_TANGLING')
    print('Finished analyze_pca', time.process_time() - start)

    # analyze_pca(DATA_PATH, DATASETS, '_AVG')
    # print('Finished analyze_pca', time.process_time() - start)

    run_dashboard(DATA_PATH, '_AVG')

else:
    analyze_pca(DATA_PATH, DATASETS)
    print('Finished analyze_pca', time.process_time() - start)

    run_dashboard(DATA_PATH)
