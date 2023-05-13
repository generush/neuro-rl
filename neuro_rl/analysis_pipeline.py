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

DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/'

DATASETS = [
    # 'OBS',
    'ACT',
    'A_MLP_XX',
    'A_LSTM_CX',
    'A_LSTM_HX',
    'A_LSTM_C1X',
    'A_LSTM_C2X',
    # 'C_MLP_XX',
    # 'C_LSTM_CX',
    # 'C_LSTM_HX',
    # 'A_GRU_HX',
    # 'C_GRU_HX',
]

AVG = True

start = time.process_time()

if AVG:
    # analyze_cycle(DATA_PATH)
    # print('Finished analyze_cycle', time.process_time() - start)

    # analyze_pca_speed_axis(DATA_PATH, DATASETS, '_AVG')
    # print('Finished analyze_pca', time.process_time() - start)

    analyze_pca(DATA_PATH, DATASETS, '_AVG')
    print('Finished analyze_pca', time.process_time() - start)

    analyze_tangling(DATA_PATH, DATASETS, '_AVG')
    print('Finished analyze_tangling', time.process_time() - start)

    run_dashboard(DATA_PATH, '_AVG')

else:
    analyze_pca(DATA_PATH, DATASETS)
    print('Finished analyze_pca', time.process_time() - start)

    run_dashboard(DATA_PATH)
