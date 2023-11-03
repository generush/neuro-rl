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

def analyze_cycle(path: str, file: str):

    # load DataFrame
    df = pd.read_parquet(path + file + '.parquet')

    # get dt time step
    DT = df['TIME'][1] - df['TIME'][0]

    # this entry is the first time step of swing phase (this FOOT_FORCE_002 = zero)
    mask_swingf = df['FT_FORCE_RAW_002'] == 0

    # the prior entry is the last time step of stance phase (the previous FOOT_FORCE_002 > 0)
    mask_swing0 = (df['FT_FORCE_RAW_002'] > 0).shift()

    # first time step entry of new condition (env id)
    mask_new = df['ENV'].diff()

    # create dataframe of all ones
    mask_ones = df['ENV'] >= 0

    # compute the cycle id (when starting swing phase or when new test)
    df.insert(loc=0, column='CYCLE_NUM', value=(mask_swingf & mask_swing0 | mask_new).cumsum())

    # time step index per cycle
    df.insert(loc=1, column='CYCLE_TIME', value=mask_ones.groupby(df['CYCLE_NUM']).cumsum() - 1)

    # get the cycle length, for each cycle
    cyc_len_df = df.groupby(['CYCLE_NUM'])['CYCLE_TIME'].max() + 1 # add one b/c index zero: period = index + 1
    df = pd.merge(df, cyc_len_df, on=['CYCLE_NUM'], how='left')
    df = df.rename(columns={'CYCLE_TIME_x': 'CYCLE_TIME'})
    df = df.rename(columns={'CYCLE_TIME_y': 'CYCLE_PERIOD'})

    # get the most common cycle length for each set of u,v commands
    mode_df = df.groupby(['CYCLE_NUM', 'OBS_RAW_009_u_star', 'OBS_RAW_010_v_star', 'OBS_RAW_011_r_star'])['CYCLE_PERIOD'].max().reset_index().groupby(['OBS_RAW_009_u_star', 'OBS_RAW_010_v_star', 'OBS_RAW_011_r_star'])['CYCLE_PERIOD'].apply(lambda x: x.mode())

    # merge the grouped DataFrame with the original DataFrame to add the new column
    df = pd.merge(df, mode_df, on=['OBS_RAW_009_u_star', 'OBS_RAW_010_v_star', 'OBS_RAW_011_r_star'], how='left')
    df = df.rename(columns={'CYCLE_PERIOD_x': 'CYCLE_PERIOD'})
    df = df.rename(columns={'CYCLE_PERIOD_y': 'Mode_of_Max_Value'})

    # create a new DataFrame that only includes data that matches the mode of the maximum value
    filtered_df = df[df['CYCLE_PERIOD'] == df['Mode_of_Max_Value']].reset_index(drop=True)

    # AVERAGE cycles based on the u, v, r commands
    avg_df = filtered_df.groupby(['CYCLE_TIME', 'OBS_RAW_009_u_star', 'OBS_RAW_010_v_star', 'OBS_RAW_011_r_star']).mean().reset_index()

    # sort by condition and then by time
    avg_sorted_df = avg_df.sort_values(['OBS_RAW_009_u_star', 'OBS_RAW_010_v_star', 'OBS_RAW_011_r_star', 'CYCLE_TIME'], ascending=[True, True, True, True]).reset_index(drop=True)

    # delete unnecessary columns
    avg_sorted_df = avg_sorted_df.drop('ENV', axis=1)
    avg_sorted_df = avg_sorted_df.drop('TIME', axis=1)
    avg_sorted_df = avg_sorted_df.drop('CYCLE_NUM', axis=1)
    avg_sorted_df = avg_sorted_df.drop('Mode_of_Max_Value', axis=1)

    # recompute actual time based on cycle_time
    avg_sorted_df.insert(loc=0, column='TIME', value=avg_sorted_df['CYCLE_TIME'] * DT)

    # The 'CONDITION' column will contain the unique indices based on commands u, v, r
    avg_sorted_df['CONDITION'] = avg_sorted_df.groupby(['OBS_RAW_009_u_star', 'OBS_RAW_010_v_star', 'OBS_RAW_011_r_star'], sort=True).ngroup()

    # export trial-averaged data (1 cycle per CONDITION) !!!
    avg_sorted_df.to_csv(path + file + '_AVG' + '.csv')

    # COVARIANCE of cycles based on the u, v, r commands
    var_df = filtered_df.groupby(['CYCLE_TIME', 'OBS_RAW_009_u_star', 'OBS_RAW_010_v_star', 'OBS_RAW_011_r_star']).var().reset_index()

    # sort by condition and then by time
    var_sorted_df = var_df.sort_values(['OBS_RAW_009_u_star', 'OBS_RAW_010_v_star', 'OBS_RAW_011_r_star', 'CYCLE_TIME'], ascending=[True, True, True, True]).reset_index(drop=True)

    # delete unnecessary columns
    var_sorted_df = var_sorted_df.drop('ENV', axis=1)
    var_sorted_df = var_sorted_df.drop('TIME', axis=1)
    var_sorted_df = var_sorted_df.drop('CYCLE_NUM', axis=1)
    var_sorted_df = var_sorted_df.drop('Mode_of_Max_Value', axis=1)

    # recompute actual time based on cycle_time
    var_sorted_df.insert(loc=0, column='TIME', value=var_sorted_df['CYCLE_TIME'] * DT)

    # The 'CONDITION' column will contain the unique indices based on commands u, v, r
    var_sorted_df['CONDITION'] = var_sorted_df.groupby(['OBS_RAW_009_u_star', 'OBS_RAW_010_v_star', 'OBS_RAW_011_r_star'], sort=True).ngroup()

    # export trial-averaged data (1 cycle per CONDITION) !!!
    var_sorted_df.to_csv(path + file + '_VAR' + '.csv')

    return avg_sorted_df
    