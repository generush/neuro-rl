# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from typing import List

import numpy as np
import pandas as pd

from analysis.analyze_pca import import_pca, import_scl

from utils.data_processing import process_data, process_csv
from plotting.generation import generate_dropdown, generate_graph
from plotting.plot import plot_scatter3_ti_tf
from embeddings.embeddings import Data, Embeddings, MultiDimensionalScalingEmbedding, PCAEmbedding, MDSEmbedding, ISOMAPEmbedding,LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding

import sklearn.preprocessing
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

import numpy as np
import pandas as pd
import pickle as pk
import dask.dataframe as dd

def append_pc_data(FOLDER_PATH, FILE_NAME, DATASETS):

    # load DataFrame
    df = pd.read_parquet(FOLDER_PATH + FILE_NAME + '.parquet')


    for idx, data_type in enumerate(DATASETS):

        # select data for PCA analysis (only raw data)
        filt_df = df.loc[:,df.columns.str.contains(data_type + '_RAW')]

        # get number of dimensions of DataFrame
        N_DIMENSIONS = len(filt_df.columns)

        # create column name
        COLUMNS = np.char.mod(data_type + '_PC_%03d', np.arange(N_DIMENSIONS))

        if N_DIMENSIONS > 0:

            pca = import_pca(FOLDER_PATH + data_type +'_PCA' + '.pkl')
            scl = import_scl(FOLDER_PATH + data_type +'_SCL' + '.pkl')

            # transform to PCA space
            filt_pc_data = pca.transform(scl.transform(filt_df.values))
            filt_pc_df = pd.DataFrame(data=filt_pc_data, columns=COLUMNS)
            
            # append pc data (concatenating along columns)
            df = pd.concat([df, filt_pc_df], axis=1)


    # NEED TO MOVE TO ANOTHER FUNCTION SO IT OPERATES ON ENTIRE RAW_DATA AND NOT AVG DATA!
    df.to_parquet(FOLDER_PATH + 'RAW_AND_PC_DATA' + '.parquet')