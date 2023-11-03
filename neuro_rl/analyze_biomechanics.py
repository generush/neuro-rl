import logging
from collections import OrderedDict

import numpy as np
import pandas as pd
import pickle as pk
import dask.dataframe as dd

import matplotlib.pyplot as plt

# **LSTM16-DIST500 4/4 steps, W/ TERRAIN ()
# lstm_model = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_15-24-12/nn/last_AnymalTerrain_ep_3200_rew_20.145746.pth')
DATA_PATH1 = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-09-07-13-56_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/'
df1 = dd.read_csv(DATA_PATH1 + 'RAW_DATA' + '.csv').compute()

# **LSTM16-NODIST 4/4 steps, W/ TERRAIN ()
DATA_PATH2 = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-09-07-14-52_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]/'
df2 = dd.read_csv(DATA_PATH2 + 'RAW_DATA' + '.csv').compute()
import pandas as pd
import dask.dataframe as dd

# Example usage for generating leg_data instances for df1 and df2

start_index = 200
end_index = 300

df1 = dd.read_csv(DATA_PATH1 + 'RAW_DATA' + '.csv').compute()
df2 = dd.read_csv(DATA_PATH2 + 'RAW_DATA' + '.csv').compute()


def generate_structured_dict(legs, joints, signals):
    structured_dict = {}
    for leg in legs:
        structured_dict[leg] = {}
        for joint in joints:
            structured_dict[leg][joint] = {}
            for signal in signals:
                structured_dict[leg][joint][signal] = {}
    return structured_dict

# Example usage
legs = ['LF', 'LH', 'RF', 'RH']
joints = ['HAA', 'HFE', 'KFE']
signals = ['pos', 'vel', 'torque']

test1 = generate_structured_dict(legs, joints, signals)

test1['LF']['HAA']['pos'] = df1['OBS_RAW_012_dof_pos_01'].values
test1['LF']['HFE']['pos'] = df1['OBS_RAW_013_dof_pos_02'].values
test1['LF']['KFE']['pos'] = df1['OBS_RAW_014_dof_pos_03'].values
test1['LH']['HAA']['pos'] = df1['OBS_RAW_015_dof_pos_04'].values
test1['LH']['HFE']['pos'] = df1['OBS_RAW_016_dof_pos_05'].values
test1['LH']['KFE']['pos'] = df1['OBS_RAW_017_dof_pos_06'].values
test1['RF']['HAA']['pos'] = df1['OBS_RAW_018_dof_pos_07'].values
test1['RF']['HFE']['pos'] = df1['OBS_RAW_019_dof_pos_08'].values
test1['RF']['KFE']['pos'] = df1['OBS_RAW_020_dof_pos_09'].values
test1['LH']['HAA']['pos'] = df1['OBS_RAW_021_dof_pos_10'].values
test1['LH']['HFE']['pos'] = df1['OBS_RAW_022_dof_pos_11'].values
test1['LH']['KFE']['pos'] = df1['OBS_RAW_023_dof_pos_12'].values
test1['LF']['HAA']['vel'] = df1['OBS_RAW_024_dof_vel_01'].values
test1['LF']['HFE']['vel'] = df1['OBS_RAW_025_dof_vel_02'].values
test1['LF']['KFE']['vel'] = df1['OBS_RAW_026_dof_vel_03'].values
test1['LH']['HAA']['vel'] = df1['OBS_RAW_027_dof_vel_04'].values
test1['LH']['HFE']['vel'] = df1['OBS_RAW_028_dof_vel_05'].values
test1['LH']['KFE']['vel'] = df1['OBS_RAW_029_dof_vel_06'].values
test1['RF']['HAA']['vel'] = df1['OBS_RAW_030_dof_vel_07'].values
test1['RF']['HFE']['vel'] = df1['OBS_RAW_031_dof_vel_08'].values
test1['RF']['KFE']['vel'] = df1['OBS_RAW_032_dof_vel_09'].values
test1['LH']['HAA']['vel'] = df1['OBS_RAW_033_dof_vel_10'].values
test1['LH']['HFE']['vel'] = df1['OBS_RAW_034_dof_vel_11'].values
test1['LH']['KFE']['vel'] = df1['OBS_RAW_035_dof_vel_12'].values
test1['LF']['HAA']['torque'] = df1['ACT_RAW_000_LF_HAA'].values
test1['LF']['HFE']['torque'] = df1['ACT_RAW_001_LF_HFE'].values
test1['LF']['KFE']['torque'] = df1['ACT_RAW_002_LF_KFE'].values
test1['LH']['HAA']['torque'] = df1['ACT_RAW_003_LH_HAA'].values
test1['LH']['HFE']['torque'] = df1['ACT_RAW_004_LH_HFE'].values
test1['LH']['KFE']['torque'] = df1['ACT_RAW_005_LH_KFE'].values
test1['RF']['HAA']['torque'] = df1['ACT_RAW_006_RF_HAA'].values
test1['RF']['HFE']['torque'] = df1['ACT_RAW_007_RF_HFE'].values
test1['RF']['KFE']['torque'] = df1['ACT_RAW_008_RF_KFE'].values
test1['LH']['HAA']['torque'] = df1['ACT_RAW_009_RH_HAA'].values
test1['LH']['HFE']['torque'] = df1['ACT_RAW_010_RH_HFE'].values
test1['LH']['KFE']['torque'] = df1['ACT_RAW_011_RH_KFE'].values
test1['LF']['ground_force'] = df1['FT_FORCE_RAW_000'].values
test1['LH']['ground_force'] = df1['FT_FORCE_RAW_001'].values
test1['RF']['ground_force'] = df1['FT_FORCE_RAW_002'].values
test1['LH']['ground_force'] = df1['FT_FORCE_RAW_003'].values


# You can access leg_data_df1 and leg_data_df2, which contain the leg_data instances for df1 and df2, respectively.

# Define flags to turn on/off different signals
plot_position = True  # Set to True to plot position signals
plot_velocity = False  # Set to True to plot velocity signals
plot_torque = False    # Set to True to plot torque signals
plot_feet_forces = True  # Set to True to plot feet forces signals

# Specify the index range to display for df2
start_index = 200
end_index = 300

# Calculate the number of rows and columns for subplots
num_columns = 3  # Specify the number of columns for subplots
num_rows = (len(specific_columns_pos) + num_columns - 1) // num_columns

# Create subplots
fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 8))
fig.subplots_adjust(hspace=0.5)  # Adjust vertical spacing between subplots

# Loop through unique conditions and plot on subplots for df1 and df2
plot_index = 0  # Initialize the plot_index for feet forces plots
for i, (column_pos, column_vel, column_torque) in enumerate(zip(specific_columns_pos, specific_columns_vel, spec_columns_torque)):
    if column_pos in df1.columns and column_torque in df1.columns:
        mean_values_pos_df1 = df1[column_pos][start_index:end_index] if plot_position else None
        mean_values_vel_df1 = df1[column_vel][start_index:end_index] if plot_velocity else None
        mean_values_torque_df1 = df1[column_torque][start_index:end_index] if plot_torque else None

        mean_values_pos_df2 = df2[column_pos][start_index:end_index] if plot_position else None
        mean_values_vel_df2 = df2[column_vel][start_index:end_index] if plot_velocity else None
        mean_values_torque_df2 = df2[column_torque][start_index:end_index] if plot_torque else None

        ax = axes[i // num_columns, i % num_columns]

        if plot_position:
            ax.plot(mean_values_pos_df1, label=f'{column_pos} (Pos) - df1')
            ax.plot(mean_values_pos_df2, label=f'{column_pos} (Pos) - df2')
        if plot_velocity:
            ax.plot(mean_values_vel_df1, label=f'{column_vel} Integral (Vel) - df1')
            ax.plot(mean_values_vel_df2, label=f'{column_vel} Integral (Vel) - df2')
        if plot_torque:
            ax.plot(mean_values_torque_df1, label=f'{column_torque} (Torque) - df1')
            ax.plot(mean_values_torque_df2, label=f'{column_torque} (Torque) - df2')

        ax.set_xlabel("Index")
        ax.set_ylabel("Value")
        ax.legend()
    else:
        print(f"Skipping columns {column_pos}, {column_vel}, {column_torque} as they do not exist in both dataframes.")


# Show the subplots
plt.show()

print('hi')