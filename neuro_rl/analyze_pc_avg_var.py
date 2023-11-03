import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

import numpy as np
import pandas as pd
import pickle as pk
import dask.dataframe as dd

import sklearn.decomposition

from analysis.analyze_pca import compute_pca

import matplotlib.pyplot as plt

# **LSTM16-DIST500 4/4 steps, W/ TERRAIN ()
# lstm_model = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_15-24-12/nn/last_AnymalTerrain_ep_3200_rew_20.145746.pth')
DATA_PATH = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-08-27-16-41_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/'

# load DataFrame
df_avg = dd.read_csv(DATA_PATH + 'RAW_AND_PC_DATA_AVG' + '.csv').compute()
df_var = dd.read_csv(DATA_PATH + 'RAW_AND_PC_DATA_VAR' + '.csv').compute()

print('hi')

# List of specific columns to plot
specific_columns = [
    'A_LSTM_HC_PC_000',
    'A_LSTM_HC_PC_001',
    'A_LSTM_HC_PC_002'
]

# Create a figure
plt.figure(figsize=(10, 6))

# Loop through unique conditions
for condition in df_avg['CONDITION'].unique():
    df_avg_condition = df_avg[df_avg['CONDITION'] == condition].reset_index(drop=True)
    df_var_condition = df_var[df_var['CONDITION'] == condition].reset_index(drop=True)
    
    for column in specific_columns:
        if column in df_avg.columns and column in df_var.columns:
            mean_values = df_avg_condition[column]
            var_values = df_var_condition[column]

            upper_bound = mean_values + 2 * np.sqrt(var_values)
            lower_bound = mean_values - 2 * np.sqrt(var_values)

            plt.plot(mean_values, label=f"Mean of {column} - {condition}")
            plt.fill_between(range(len(mean_values)), lower_bound, upper_bound, alpha=0.5, label=f"Variance of {column} - {condition}")

        else:
            print(f"Skipping {column} as it does not exist in both dataframes.")

# Set plot labels and legend
plt.title(f"Mean and Variance of Specific Columns for Different Conditions")
plt.xlabel("Index")
plt.ylabel("Value")
plt.legend()

# Show the plot
plt.show()

print('hi')