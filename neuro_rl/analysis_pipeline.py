# %%


import pandas as pd

from cfg.loader import load_configuration
from utils.data_processing import filter_by_column_keywords
from analysis.compute_avg_gait_cycle import compute_avg_gait_cycle
from analysis.analyze_traj import analyze_traj
from analysis.append_pc import append_pc
from analysis.append_speed_axis import append_speed_axis
from analysis.compute_interpolation import compute_interpolation
from analysis.plot_pc12_speed_axis import plot_pc12_speed_axis
from analysis.append_tangling import append_tangling
from analysis.create_pc_data import create_pc_data
from plotting.dashboard import run_dashboard

# %matplotlib widget

from cfg.loader import load_configuration

cfg = load_configuration()

# %% Load DataFrame
raw_df = pd.read_parquet(cfg.data_paths.input + 'RAW_DATA' + '.parquet')

# %% Filter DataFrame
filt_df = filter_by_column_keywords(raw_df, cfg.dataset_names, 'RAW')

# %% Compute cycle-average and variance datasets from raw dataset
avg_cycle_df, var_cycle_df = compute_avg_gait_cycle(filt_df)

# %% Append pc data
avg_cycle_df, pca_dict = append_pc(avg_cycle_df, cfg.dataset_names, cfg.max_num_principal_components, cfg.normalization_type)

# %% Append tangling data
avg_cycle_df = append_tangling(avg_cycle_df, cfg.dataset_names, cfg.tangling_type)

# %% Append speed axis
avg_cycle_df = append_speed_axis(avg_cycle_df, cfg.dataset_names, cfg.max_num_principal_components, cfg.normalization_type)

# %% Compute interpolated dataset from cycle-average dataset
avg_cycle_interp_df = compute_interpolation(avg_cycle_df)

# %% Plot ppeed axis figures
plot_pc12_speed_axis(avg_cycle_interp_df, cfg.dataset_names)

# %%

print('hi')
