# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from typing import List

import numpy as np
import pandas as pd

from utils.data_processing import process_data
from plotting.generation import generate_dropdown, generate_graph
from plotting.plot import plot_scatter3_ti_tf
from embeddings.embeddings import Data, Embeddings, MultiDimensionalScalingEmbedding, PCAEmbedding, MDSEmbedding, ISOMAPEmbedding,LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding

import sklearn.preprocessing
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

def compute_pca(df_raw, n_components, columns):


    # # get column names that contain the string "RAW"
    # cols_to_normalize = [col for col in df_raw.columns if 'RAW' in col]

    # # normalize only the columns that contain the string "norm", avoiding normalization if max = min
    # col_min = df_raw[cols_to_normalize].min()
    # col_max = df_raw[cols_to_normalize].max()
    # col_range = (col_max - col_min).compute() + 5
    # col_range[col_range == 0] = 1 # avoid division by zero
    # normalized_df = df_raw[cols_to_normalize] / col_range

    # # concatenate the normalized dataframe with the original dataframe along the columns axis
    # df_raw = pd.concat([df_raw.drop(cols_to_normalize, axis=1).compute(), normalized_df.compute()], axis=1)

    # added scaling since dimensions might have different magnitudes
    
    scaler = sklearn.preprocessing.StandardScaler()
    df_scaled = scaler.fit_transform(df_raw)

    # create PCA object
    pca = sklearn.decomposition.PCA(n_components=n_components)

    # fit pca transform
    data_pc = pca.fit_transform(df_scaled)

    # create DataFrame
    df_pc = pd.DataFrame(data_pc)

    # name DataFrame columns
    df_pc.columns = columns

    return pca, df_pc

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk

def export_pca(pca: sklearn.decomposition.PCA, path: str):
    pk.dump(pca, open(path,"wb"))

def import_pca(path: str):
    return pk.load(open(path,'rb'))

def analyze_pca(path: str, data_names: List[str], file_suffix: str = ''):

    N_COMPONENTS = 10

    # load DataFrame
    data = process_data(path + 'NORM_DATA' + file_suffix + '.csv')

    for idx, data_type in enumerate(data_names):

        # select data for PCA analysis (only raw data)
        filt_data = data.loc[:,data.columns.str.contains(data_type + '_RAW')]

        # get number of dimensions of DataFrame
        N_DIMENSIONS = len(filt_data.columns)

        # create column name
        COLUMNS = np.char.mod(data_type + '_PC_%03d', np.arange(N_COMPONENTS))

        if N_DIMENSIONS > 0:

            # computa pca
            pca, pc_df = compute_pca(filt_data, N_COMPONENTS, COLUMNS)

            # export PCA object
            export_pca(pca, path + data_type +'_PCA' + '.pkl')

            # export DataFrame
            pc_df.to_csv(path + data_type + '_' + 'PC_DATA' + file_suffix + '.csv')

