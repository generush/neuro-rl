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

######################## Loading all data files ##########################

# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/AnymalTerrain_2023-03-14_17-28-09/'
# DATA_PATH = '/home/gene/code/rl_neural_dynamics/IsaacGymEnvs/isaacgymenvs/videos/ShadowHandAsymmLSTM_2023-03-14_16-19-22/'

# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/shadowhand_2023_03_11_1279/'
# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/anymalterrain_2023_04_17_00/'
# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/anymalterrain_2023_04_17_01/'
# DATA_PATH = '/home/gene/code/neuro-rl/IsaacGymEnvs/isaacgymenvs/anymalterrain_2023_04_17_AGENT_17_44/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data_AnymalTerrain_Flat_t0_t1000/'
DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/'

# load DataFrame
df = pd.read_csv(DATA_PATH + 'RAW_DATA_ALL' + '.csv', index_col=0)

# this entry is the first time step of swing phase (this FOOT_FORCE_002 = zero)
mask_swingf = df['FOOT_FORCES_002'] == 0

# the prior entry is the last time step of stance phase (the previous FOOT_FORCE_002 > 0)
mask_swing0 = (df['FOOT_FORCES_002'] > 0).shift()

# first time step entry of new condition
mask_new = df['CONDITION'].diff()

# create dataframe one all ones
mask_ones = df['CONDITION'] >= 0

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
mode_df = df.groupby(['CYCLE_NUM', 'OBS_RAW_009_u_star', 'OBS_RAW_010_v_star'])['CYCLE_PERIOD'].max().reset_index().groupby(['OBS_RAW_009_u_star', 'OBS_RAW_010_v_star'])['CYCLE_PERIOD'].apply(lambda x: x.mode())

# merge the grouped DataFrame with the original DataFrame to add the new column
df = pd.merge(df, mode_df, on=['OBS_RAW_009_u_star', 'OBS_RAW_010_v_star'], how='left')
df = df.rename(columns={'CYCLE_PERIOD_x': 'CYCLE_PERIOD'})
df = df.rename(columns={'CYCLE_PERIOD_y': 'Mode_of_Max_Value'})

# create a new DataFrame that only includes data that matches the mode of the maximum value
filtered_df = df[df['CYCLE_PERIOD'] == df['Mode_of_Max_Value']]

# average cycles based on the u,v commands
avg_df = filtered_df.groupby(['CYCLE_TIME', 'OBS_RAW_009_u_star', 'OBS_RAW_010_v_star']).mean().reset_index()

# sort by condition and then by time
avg_sorted_df = avg_df.sort_values(['CONDITION', 'CYCLE_TIME'], ascending=[True, True]).reset_index(drop=True)

# delete unnecessary columns
avg_sorted_df = avg_sorted_df.drop('TIME', axis=1)
avg_sorted_df = avg_sorted_df.drop('CYCLE_NUM', axis=1)
# avg_sorted_df = avg_sorted_df.drop('CYCLE_PERIOD', axis=1)
avg_sorted_df = avg_sorted_df.drop('Mode_of_Max_Value', axis=1)
# avg_sorted_df = avg_sorted_df.drop('FOOT_FORCES_000', axis=1)
# avg_sorted_df = avg_sorted_df.drop('FOOT_FORCES_001', axis=1)
# avg_sorted_df = avg_sorted_df.drop('FOOT_FORCES_002', axis=1)
# avg_sorted_df = avg_sorted_df.drop('FOOT_FORCES_003', axis=1)

# recompute actual time based on cycle_time
avg_sorted_df.insert(loc=0, column='TIME', value=avg_sorted_df['CYCLE_TIME'] * 0.004)
avg_sorted_df.to_csv(DATA_PATH + 'RAW_DATA' + '.csv')

print('hi')