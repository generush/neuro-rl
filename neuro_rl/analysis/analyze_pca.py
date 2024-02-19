# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from typing import List

import numpy as np
import pandas as pd
import dask.dataframe as dd

from utils.data_processing import process_data, process_csv
from plotting.generation import generate_dropdown, generate_graph
from plotting.plot import plot_scatter3_ti_tf
from embeddings.embeddings import Data, Embeddings, MultiDimensionalScalingEmbedding, PCAEmbedding, MDSEmbedding, ISOMAPEmbedding,LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding

import sklearn.preprocessing
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics
from constants.normalization_types import NormalizationType
from utils.scalers import RangeScaler, RangeSoftScaler

def compute_pca(df_raw, n_components, norm_type: NormalizationType, columns):

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

    # Example usage
    if norm_type == NormalizationType.Z_SCORE.value:
        # Perform z-score normalization

        scl = sklearn.preprocessing.StandardScaler()

        df_scaled = scl.fit_transform(df_raw)
        
    elif norm_type == NormalizationType.RANGE.value:
        # Perform range normalization
        
        scl = RangeScaler()
        df_scaled = scl.fit_transform(df_raw)
        
    elif norm_type == NormalizationType.RANGE_SOFT.value:
        # Perform range soft normalization
        
        scl = RangeSoftScaler(softening_factor=5)  # You can adjust the softening factor as needed
        df_scaled = scl.fit_transform(df_raw)

    else:

        raise ValueError(f"Unsupported normalization type: {norm_type}")
        

    # create PCA object
    pca = sklearn.decomposition.PCA(n_components=n_components)

    # fit pca transform
    data_pc = pca.fit_transform(df_scaled)

    # create DataFrame
    df_pc = pd.DataFrame(data_pc)

    # name DataFrame columns
    df_pc.columns = columns

    return scl, pca, df_pc

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk


def export_scl(scl: sklearn.preprocessing.StandardScaler, path: str):
    pk.dump(scl, open(path,"wb"))

def export_pca(pca: sklearn.decomposition.PCA, path: str):
    pk.dump(pca, open(path,"wb"))

def import_scl(path: str):
    return pk.load(open(path,'rb'))

def import_pca(path: str):
    return pk.load(open(path,'rb'))

def analyze_pca(df: pd.DataFrame, data_names: List[str], max_dims: int, norm_type: NormalizationType, file_suffix: str = ''):

    # load DataFrame
    # data = process_csv(path + file + file_suffix + '.csv')
    # data['TIME']

    # meta_data = data.loc[:,~data.columns.str.contains('_RAW')].compute()
    # meta_data.to_csv(path + 'META_DATA' + file_suffix + '.csv')

    # Create a dictionary to store scl and pca objects
    pca_dict = {}

    for idx, data_type in enumerate(data_names):

        # select data for PCA analysis (only raw data)
        filt_data = df.loc[:,df.columns.str.contains(data_type + '_RAW')]

        # get number of dimensions of DataFrame
        n_dims = min(len(filt_data.columns), max_dims)

        # create column name
        COLUMNS = np.char.mod(data_type + '_PC_%03d', np.arange(n_dims))

        if n_dims > 0:

            # compute pca
            scl, pca, pc_df = compute_pca(filt_data, n_dims, norm_type, COLUMNS)

            # export PCA object
            export_pca(pca, data_type +'_PCA' + '.pkl')
            export_scl(scl, data_type +'_SCL' + '.pkl')

            # Store scl and pca objects in the dictionary
            pca_dict[data_type] = {'scl': scl, 'pca': pca}

            # export PC DataFrame
            pc_df.to_csv(data_type + '_' + 'PC_DATA' + file_suffix + '.csv')

            # Define a function to concatenate pc_df to each partition of filt_data
            def concat_partitions(partition, pc_df):
                # Convert the partition (which is a Pandas DataFrame) to the same index as pc_df if needed
                return partition.join(pc_df, how='left')

            # # Use map_partitions to apply the function to each partition of filt_data
            # df = df.map_partitions(concat_partitions, pc_df=pc_df)

            # Append pc_df to filt_data
            df = pd.concat([df, pc_df], axis=1)

            print('hello')

    return df, pca_dict

