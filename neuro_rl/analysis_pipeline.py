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
