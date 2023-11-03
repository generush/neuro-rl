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

import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

def analyze_traj(path: str):

    # load DataFrame
    df = pd.read_parquet(path + 'RAW_DATA' + '.parquet')

    # average data across different time steps
    avg_df = df.groupby('TIME').mean().reset_index()
    
    # remove env column
    avg_df = avg_df.drop('ENV', axis=1)

    # export trial-averaged data (1 cycle per CONDITION) !!!
    avg_df.to_csv(path + 'RAW_DATA_AVG' + '.csv')




    return avg_df