# %%


import pandas as pd

from utils.data_processing import filter_by_column_keywords
from analysis.compute_avg_gait_cycle import compute_avg_gait_cycle
from analysis.analyze_traj import analyze_traj
from analysis.append_pc import append_pc
from analysis.append_pc_speed_axis import append_pc_speed_axis
from analysis.compute_interpolation import compute_interpolation
from analysis.plot_pc12_speed_axis import plot_pc12_speed_axis
from analysis.append_tangling import append_tangling
from analysis.create_pc_data import create_pc_data
from plotting.dashboard import run_dashboard

import yaml

# %matplotlib widget

# Load YAML config file
with open('cfg/analyze/analysis.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Accessing config variables
input_data_path = config['data_paths']['input']
output_data_path = config['data_paths']['output']
dataset_names = config['dataset_names']
norm_type = config['normalization_type']
max_components = config['max_num_principal_components']
tangling_type = config['tangling_type']

# %% Load DataFrame
raw_df = pd.read_parquet(input_data_path + 'RAW_DATA' + '.parquet')

# %% Load DataFrame
filt_df = filter_by_column_keywords(raw_df, dataset_names, 'RAW')

# %% Compute cycle-average and variance datasets from raw dataset
avg_cycle_df, var_cycle_df = compute_avg_gait_cycle(filt_df)

# %% Append pc data
avg_cycle_df, pca_dict = append_pc(avg_cycle_df, dataset_names, max_components, norm_type)

# %% Append tangling data
avg_cycle_df = append_tangling(avg_cycle_df, dataset_names, tangling_type)

# %% Append speed axis
avg_cycle_df = append_pc_speed_axis(avg_cycle_df, dataset_names, max_components, norm_type)

# %% Compute interpolated dataset from cycle-average dataset
avg_cycle_interp_df = compute_interpolation(avg_cycle_df)

# %% Compute speed axis

# %% Plotting Speed Axis Figures
plot_pc12_speed_axis(avg_cycle_interp_df, dataset_names)

# %%

print('hi')
