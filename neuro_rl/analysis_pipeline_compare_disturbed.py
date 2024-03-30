import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from cfg.loader import load_configuration
def find_segments(mask):
    # Compute the difference and fill the NaN value at the beginning with False
    diff = mask.astype(int).diff().fillna(0)

    # Starts of segments are where diff is 1 (indicating a change from False to True)
    starts = diff[diff == 1].index

    # Ends of segments are where diff is -1 (indicating a change from True to False)
    ends = diff[diff == -1].index

    # If the series starts with a True, add the first index as the start of the first segment
    if mask.iloc[0]:
        starts = starts.insert(0, mask.index[0])

    # If the series ends with a True, add the last index as the end of the last segment
    if mask.iloc[-1]:
        ends = ends.append(pd.Index([mask.index[-1]]))

    # Ensure that starts and ends have the same length by trimming or padding if necessary
    # This should be rare but might occur in edge cases or with unusual data
    min_length = min(len(starts), len(ends))
    starts, ends = starts[:min_length], ends[:min_length]

    return starts, ends
def precompute_segments(df, contact_fields):
    segments_info = {}
    for field in contact_fields:
        non_nan_mask = df[field].notna()
        starts, ends = find_segments(non_nan_mask)
        segments_info[field] = {'starts': starts, 'ends': ends}
    return segments_info

def overlay_plot_signal(fields_and_nicknames, dfs_collection, segments_collections, colors, fig_width, fig_height, additional_features=None, plot_name="default_plot_name"):

    # Determine the global min and max of the 'TIME_RAW' column across all DataFrames
    global_min_time = min(df['TIME_RAW'].min() for df in dfs_collection.values())
    global_max_time = max(df['TIME_RAW'].max() for df in dfs_collection.values())

    # Determine the number of subplots needed for the current group
    num_subplots = len(fields_and_nicknames)

    # Create the figure and subplots, sharing the x-axis
    fig, axes = plt.subplots(num_subplots, 1, sharex=True, figsize=(fig_width, num_subplots * fig_height), constrained_layout=True)

    # Ensure 'axes' is an iterable (important if num_subplots is 1)
    if num_subplots == 1:
        axes = [axes]

    for ax, (x_field, y_field, nickname) in zip(axes, fields_and_nicknames):

        for df_idx, df in dfs_collection.items():

            if y_field in df.columns:
                if any(key in y_field for key in segments_collections[0].keys()):
                    ax.plot(df[x_field], df[y_field], label=f'DF{df_idx}', color=colors[df_idx], linewidth=0.5)

                    matching_keys = [key for key in segments_collections[df_idx].keys() if key in y_field]
                    if matching_keys:
                        for key in matching_keys:
                            segments_info = segments_collections[df_idx]
                            for start, end in zip(segments_info[key]['starts'], segments_info[key]['ends']):
                                ax.plot(df.loc[start:end, x_field], df.loc[start:end, y_field], color=colors[df_idx], linewidth=2)
                else:
                    ax.plot(df[x_field], df[y_field], label=f'DF{df_idx}', color=colors[df_idx], linewidth=1)
                
        if additional_features and additional_features.get('draw_perturb_shaded_box', False):

            # Find the start and end of the chunk where PERTURB_RAW=1
            if 'PERTURB_RAW' in df.columns:
                perturb_start_idx = df.index[df['PERTURB_RAW'] == 1].min()
                perturb_end_idx = df.index[df['PERTURB_RAW'] == 1].max()

                # Extend perturb_end_idx by one step, if it's not the last index
                if perturb_start_idx > df.index[0]:
                    perturb_start_idx -= 1

                # Convert indices back to TIME_RAW values
                perturb_start = df.loc[perturb_start_idx, 'TIME_RAW']
                perturb_end = df.loc[perturb_end_idx, 'TIME_RAW']

                # Check if there is a perturb chunk in the DataFrame
                if pd.notna(perturb_start) and pd.notna(perturb_end):
                    ax.axvspan(perturb_start, perturb_end, facecolor='lightgray', alpha=0.5)  # Translucent region

        if additional_features and additional_features.get('draw_centerline', False):
            # Draw a centerline
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylabel(nickname, rotation=0)
    axes[-1].set_xlim([global_min_time, global_max_time])  # Set the same x-axis range for all subplots
    axes[-1].set_xlabel('Time [s]')
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    fig.savefig(f"{plot_name}.svg", format='svg', dpi=600)

def overlay_plot_gait(fields_and_nicknames, dfs_collection, segments_collections, colors, fig_width, fig_height, additional_features=None, plot_name="default_plot_name"):
    
    # Determine the global min and max of the 'TIME_RAW' column across all DataFrames
    global_min_time = min(df['TIME_RAW'].min() for df in dfs_collection.values())
    global_max_time = max(df['TIME_RAW'].max() for df in dfs_collection.values())

    # Determine the number of subplots needed for the current group
    num_subplots = len(dfs_collection)

    # Create the figure and subplots, sharing the x-axis
    fig, axes = plt.subplots(num_subplots, 1, sharex=True, figsize=(fig_width, num_subplots * fig_height), constrained_layout=True)

    # Ensure 'axes' is an iterable (important if num_subplots is 1)
    if num_subplots == 1:
        axes = [axes]
    
    # Iterate over each DataFrame in the collection
    for ax, (df_key, df) in zip(axes, dfs_collection.items()):

        # Iterate over each signal field you want to plot for this DataFrame
        for field_idx, (x_field, y_field, nickname) in enumerate(fields_and_nicknames):
            if y_field in df.columns:
                if any(key in y_field for key in segments_collections[0].keys()):
                    ax.plot(df[x_field], df[y_field], label=nickname, color=colors[field_idx], linewidth=0.5)

                # Check for matching segment keys
                matching_keys = [key for key in segments_collections[df_key].keys() if key in y_field]

                if matching_keys:
                    for key in matching_keys:
                        segments_info = segments_collections[df_key]
                        for start, end in zip(segments_info[key]['starts'], segments_info[key]['ends']):
                            ax.plot(df.loc[start:end, x_field], df.loc[start:end, y_field], color=colors[field_idx], linewidth=2)
                else:
                    ax.plot(df[x_field], df[y_field], label=nickname, color=colors[field_idx], linewidth=1)

        if additional_features and additional_features.get('draw_perturb_shaded_box', False):

            # Find the start and end of the chunk where PERTURB_RAW=1
            if 'PERTURB_RAW' in df.columns:
                perturb_start_idx = df.index[df['PERTURB_RAW'] == 1].min()
                perturb_end_idx = df.index[df['PERTURB_RAW'] == 1].max()

                # Extend perturb_end_idx by one step, if it's not the last index
                if perturb_start_idx > df.index[0]:
                    perturb_start_idx -= 1

                # Convert indices back to TIME_RAW values
                perturb_start = df.loc[perturb_start_idx, 'TIME_RAW']
                perturb_end = df.loc[perturb_end_idx, 'TIME_RAW']

                # Check if there is a perturb chunk in the DataFrame
                if pd.notna(perturb_start) and pd.notna(perturb_end):
                    ax.axvspan(perturb_start, perturb_end, facecolor='lightgray', alpha=0.5)  # Translucent region
                
        if additional_features and additional_features.get('draw_centerline', False):
            # Draw a centerline
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax.set_ylabel(f'DF{df_key}')
    if x_field == 'TIME_RAW':
        axes[-1].set_xlim([global_min_time, global_max_time])  # Set the same x-axis range for all subplots
        axes[-1].set_xlabel('Time [s]')
    else:
        axes[-1].set_xlabel(x_field)
        # Set the aspect ratio to be equal
        axes[-1].set_aspect('equal', adjustable='box')
    axes[-1].legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    fig.savefig(f"{plot_name}.svg", format='svg', dpi=600)
        
# def plot_gradient_heatmaps(dfs_collection, column_prefix, num_gradients=128, global_time_window=(-1, 2.5)):
#     # Initialize an empty list to store the heatmap data for each gradient
#     heatmap_data_list = []

#     # Prepare the custom colormap with white at the center
#     seismic = plt.cm.get_cmap('seismic', 256)
#     newcolors = seismic(np.linspace(0, 1, 256))
#     white = np.array([1, 1, 1, 1])
#     newcolors[128:129, :] = white
#     custom_cmap = LinearSegmentedColormap.from_list('CustomSeismic', newcolors)

#     # Filter the DataFrame according to the global_time_window and extract relevant gradient columns
#     df = dfs_collection[0][(dfs_collection[0]['TIME_RAW'] >= global_time_window[0]) & (dfs_collection[0]['TIME_RAW'] <= global_time_window[1])]
#     for i in range(num_gradients):
#         column_name = f'{column_prefix}{str(i).zfill(3)}'
#         heatmap_data_list.append(df[column_name].values)

#     # Create subplots
#     fig, axes = plt.subplots(num_gradients, 1, sharex=True, figsize=(10, 2*num_gradients), constrained_layout=True)

#     # Plot each gradient dimension as a separate subplot
#     for i, ax in enumerate(axes):
#         heatmap_data = heatmap_data_list[i].reshape(1, -1)
#         pos = ax.imshow(heatmap_data, aspect='auto', cmap=custom_cmap, norm=Normalize(vmin=-np.max(np.abs(heatmap_data)), vmax=np.max(np.abs(heatmap_data))))
#         ax.set_ylabel(f'Grad {i}', rotation=0, labelpad=20)
#         ax.get_yaxis().set_visible(False)

#         # Draw perturbation shading
#         if 'PERTURB_RAW' in df.columns:
#             perturb_start_idx = np.where(df['PERTURB_RAW'].values == 1)[0].min()
#             perturb_end_idx = np.where(df['PERTURB_RAW'].values == 1)[0].max()
#             perturb_start_time = df.iloc[perturb_start_idx]['TIME_RAW']
#             perturb_end_time = df.iloc[perturb_end_idx]['TIME_RAW']
#             ax.axvspan(perturb_start_time, perturb_end_time, facecolor='lightgray', alpha=0.5)

#     # Set the x-axis label on the last subplot
#     axes[-1].set_xlabel('Time [s]')
#     fig.colorbar(pos, ax=axes.ravel().tolist(), label='Gradient Value')

#     plt.show()
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

def overlay_plot_gradient_heatmaps(fields_and_nicknames, dfs_collection, segments_collections, colors, fig_width, fig_height, additional_features=None, plot_name="default_plot_name"):
    # Determine the number of subplots needed (one for each DataFrame in the collection)
    num_subplots = len(dfs_collection)

    # Create the figure and subplots, sharing the x-axis
    fig, axes = plt.subplots(num_subplots, 1, sharex=True, figsize=(fig_width, num_subplots * fig_height), constrained_layout=True)

    # Ensure 'axes' is an iterable (important if num_subplots is 1)
    if num_subplots == 1:
        axes = [axes]

    # Iterate over each DataFrame in the collection
    for ax, (df_key, df) in zip(axes, dfs_collection.items()):
                    
        # Assuming 'df' is the current DataFrame you are plotting
        time_values = df['TIME_RAW'].values

        # Initialize an empty list to collect gradient arrays
        gradient_arrays = []

        # Iterate over each gradient dimension and collect it
        for _, y_field, _ in fields_and_nicknames:
            gradient_arrays.append(df[y_field].to_numpy())

        # Stack collected gradients to form a T x 128 array
        gradient_data = np.stack(gradient_arrays, axis=1)  # This assumes len(fields_and_nicknames) == 128

        # Plot the heatmap
        im = ax.imshow(np.transpose(gradient_data), aspect='auto', cmap='seismic', norm=Normalize(vmin=-np.max(np.abs(gradient_data)), vmax=np.max(np.abs(gradient_data))))

        # Now, adjust the x-axis to reflect the 'TIME_RAW' values
        # Choose an interval for the ticks based on your preference and the density of your time data
        tick_interval = 50  # for example, setting a tick every 50th time step
        ax.set_xticks(np.arange(0, len(time_values), tick_interval))
        ax.set_xticklabels([f'{time:.2f}' for time in time_values[::tick_interval]], rotation=45, ha='right')

        # Draw perturbation shading if requested
        if additional_features and additional_features.get('draw_perturb_shaded_box', False) and 'PERTURB_RAW' in df.columns:
            perturb_mask = df['PERTURB_RAW'] == 1
            if perturb_mask.any():
                perturb_start, perturb_end = np.where(perturb_mask)[0][[0, -1]]
                ax.axvspan(perturb_start, perturb_end, facecolor='lightgray', alpha=0.5)

        ax.set_title(f'Gradients for DF{df_key}')
        ax.set_ylabel('Gradient Index')

        # When there's only one subplot, pass 'ax' directly
        fig.colorbar(im, ax=ax, label='Gradient Value')
    
    fig.savefig(f"{plot_name}.svg", format='svg', dpi=600)

def run_analysis():
    cfg = load_configuration()
    dfs_collection = {}
    segments_collections = {}
    contact_fields = ['LF', 'LH', 'RH', 'RF']
    colors = [
    'k',  # Black
    'b',  # Blue
    'g',  # Green
    'r',  # Red
    'm',  # Magenta
    '#ff7f0e',  # Safety Orange
    '#2ca02c',  # Cooked Asparagus Green, different shade from 'g'
    '#9467bd',  # Muted Purple
    '#8c564b',  # Chestnut Brown
    '#e377c2',  # Raspberry Yogurt Pink
    '#7f7f7f',  # Middle Gray
    '#17becf',  # Blue-Teal, different shade from 'b'
]

    for idx, (data_path, model_path, output_path) in enumerate(zip(cfg.data_path, cfg.model_path, cfg.output_path)):

        print(data_path, model_path, output_path)

        # Load DataFrame
        raw_df = pd.read_parquet(data_path + 'RAW_DATA' + '.parquet')
                
        # Find the first index where DONE_RAW == 1
        first_done_index = raw_df['DONE_RAW'].idxmax()

        # Check if DONE_RAW == 1 exists, to avoid setting everything to NaN if there are no 1's
        if raw_df.loc[first_done_index, 'DONE_RAW'] == 1:
            # Set all values to NaN for all columns beyond the first index where DONE_RAW == 1
            raw_df.loc[first_done_index + 1:] = np.nan      

        # Convert to deg
        raw_df['OBS_RAW_012_dof_pos_angle_deg_01_LF_HAA'] = 180 / np.pi * (raw_df['OBS_RAW_012_dof_pos_01'])
        raw_df['OBS_RAW_013_dof_pos_angle_deg_02_LF_HFE'] = 180 / np.pi * (raw_df['OBS_RAW_013_dof_pos_02'])
        raw_df['OBS_RAW_014_dof_pos_angle_deg_03_LF_KFE'] = 180 / np.pi * (raw_df['OBS_RAW_014_dof_pos_03'])
        raw_df['OBS_RAW_015_dof_pos_angle_deg_04_LH_HAA'] = 180 / np.pi * (raw_df['OBS_RAW_015_dof_pos_04'])
        raw_df['OBS_RAW_016_dof_pos_angle_deg_05_LH_HFE'] = 180 / np.pi * (raw_df['OBS_RAW_016_dof_pos_05'])
        raw_df['OBS_RAW_017_dof_pos_angle_deg_06_LH_KFE'] = 180 / np.pi * (raw_df['OBS_RAW_017_dof_pos_06'])
        raw_df['OBS_RAW_018_dof_pos_angle_deg_07_RF_HAA'] = 180 / np.pi * (raw_df['OBS_RAW_018_dof_pos_07'])
        raw_df['OBS_RAW_019_dof_pos_angle_deg_08_RF_HFE'] = 180 / np.pi * (raw_df['OBS_RAW_019_dof_pos_08'])
        raw_df['OBS_RAW_020_dof_pos_angle_deg_09_RF_KFE'] = 180 / np.pi * (raw_df['OBS_RAW_020_dof_pos_09'])
        raw_df['OBS_RAW_021_dof_pos_angle_deg_10_RH_HAA'] = 180 / np.pi * (raw_df['OBS_RAW_021_dof_pos_10'])
        raw_df['OBS_RAW_022_dof_pos_angle_deg_11_RH_HFE'] = 180 / np.pi * (raw_df['OBS_RAW_022_dof_pos_11'])
        raw_df['OBS_RAW_023_dof_pos_angle_deg_12_RH_KFE'] = 180 / np.pi * (raw_df['OBS_RAW_023_dof_pos_12'])

        raw_df['OBS_RAW_003_p_deg_s'] = 180 / np.pi * (raw_df['OBS_RAW_003_p'])
        raw_df['OBS_RAW_004_q_deg_s'] = 180 / np.pi * (raw_df['OBS_RAW_004_q'])
        raw_df['OBS_RAW_005_r_deg_s'] = 180 / np.pi * (raw_df['OBS_RAW_005_r'])

        # Compute OBS_RAW_007_theta
        raw_df['OBS_RAW_006_phi_angle_deg'] = 180 / np.pi * np.arcsin(raw_df['OBS_RAW_007_theta_proj'])
        raw_df['OBS_RAW_007_theta_angle_deg'] = 180 / np.pi * np.arcsin(raw_df['OBS_RAW_006_phi_proj'])
        raw_df['COM_YAW_RAW_angle_deg'] = 180 / np.pi * np.arcsin(raw_df['COM_YAW_RAW'])
        
        # Find the index where PERTURB == 1 and align the TIME column based on this index
        perturb_index = raw_df[raw_df['PERTURB_BEGIN_RAW'] == 1].index[0]
        raw_df['TIME_RAW'] -= raw_df.loc[perturb_index, 'TIME_RAW']

        # Add columns for foot contacts
        raw_df['LF'] = np.where(raw_df['FT_FORCE_RAW_000'] > 0, 4, np.nan)
        raw_df['LH'] = np.where(raw_df['FT_FORCE_RAW_001'] > 0, 3, np.nan)
        raw_df['RF'] = np.where(raw_df['FT_FORCE_RAW_002'] > 0, 1, np.nan)
        raw_df['RH'] = np.where(raw_df['FT_FORCE_RAW_003'] > 0, 2, np.nan)

        # Create new columns with LF, LH, RH, RF in column name
        raw_df['FT_X_RAW_000_LF'] = raw_df['FT_X_RAW_000']
        raw_df['FT_X_RAW_001_LH'] = raw_df['FT_X_RAW_001']
        raw_df['FT_X_RAW_002_RF'] = raw_df['FT_X_RAW_002']
        raw_df['FT_X_RAW_003_RH'] = raw_df['FT_X_RAW_003']

        # Create new columns with LF, LH, RH, RF in column name
        raw_df['FT_Y_RAW_000_LF'] = raw_df['FT_Y_RAW_000']
        raw_df['FT_Y_RAW_001_LH'] = raw_df['FT_Y_RAW_001']
        raw_df['FT_Y_RAW_002_RF'] = raw_df['FT_Y_RAW_002']
        raw_df['FT_Y_RAW_003_RH'] = raw_df['FT_Y_RAW_003']
        
        # Create new columns with LF, LH, RH, RF in column name
        raw_df['FT_Z_RAW_000_LF'] = raw_df['FT_Z_RAW_000']
        raw_df['FT_Z_RAW_001_LH'] = raw_df['FT_Z_RAW_001']
        raw_df['FT_Z_RAW_002_RF'] = raw_df['FT_Z_RAW_002']
        raw_df['FT_Z_RAW_003_RH'] = raw_df['FT_Z_RAW_003']
        
        # Create new columns with LF, LH, RH, RF in column name
        raw_df['FT_FORCE_RAW_000_LF'] = raw_df['FT_FORCE_RAW_000']
        raw_df['FT_FORCE_RAW_001_LH'] = raw_df['FT_FORCE_RAW_001']
        raw_df['FT_FORCE_RAW_002_RF'] = raw_df['FT_FORCE_RAW_002']
        raw_df['FT_FORCE_RAW_003_RH'] = raw_df['FT_FORCE_RAW_003']

        # cos(psi) - sin(psi)
        raw_df['FT_X_RAW_000_LF_body'] = (raw_df['FT_X_RAW_000_LF'] - raw_df['COM_X_RAW']) * np.cos(-raw_df['COM_YAW_RAW']) - (raw_df['FT_Y_RAW_000_LF'] - raw_df['COM_Y_RAW']) * np.sin(-raw_df['COM_YAW_RAW'])
        raw_df['FT_X_RAW_001_LH_body'] = (raw_df['FT_X_RAW_001_LH'] - raw_df['COM_X_RAW']) * np.cos(-raw_df['COM_YAW_RAW']) - (raw_df['FT_Y_RAW_001_LH'] - raw_df['COM_Y_RAW']) * np.sin(-raw_df['COM_YAW_RAW'])
        raw_df['FT_X_RAW_002_RF_body'] = (raw_df['FT_X_RAW_002_RF'] - raw_df['COM_X_RAW']) * np.cos(-raw_df['COM_YAW_RAW']) - (raw_df['FT_Y_RAW_002_RF'] - raw_df['COM_Y_RAW']) * np.sin(-raw_df['COM_YAW_RAW'])
        raw_df['FT_X_RAW_003_RH_body'] = (raw_df['FT_X_RAW_003_RH'] - raw_df['COM_X_RAW']) * np.cos(-raw_df['COM_YAW_RAW']) - (raw_df['FT_Y_RAW_003_RH'] - raw_df['COM_Y_RAW']) * np.sin(-raw_df['COM_YAW_RAW'])

        # sin(psi) + cos(psi)
        raw_df['FT_Y_RAW_000_LF_body'] = (raw_df['FT_X_RAW_000_LF'] - raw_df['COM_X_RAW']) * np.sin(-raw_df['COM_YAW_RAW']) + (raw_df['FT_Y_RAW_000_LF'] - raw_df['COM_Y_RAW']) * np.cos(-raw_df['COM_YAW_RAW'])
        raw_df['FT_Y_RAW_001_LH_body'] = (raw_df['FT_X_RAW_001_LH'] - raw_df['COM_X_RAW']) * np.sin(-raw_df['COM_YAW_RAW']) + (raw_df['FT_Y_RAW_001_LH'] - raw_df['COM_Y_RAW']) * np.cos(-raw_df['COM_YAW_RAW'])
        raw_df['FT_Y_RAW_002_RF_body'] = (raw_df['FT_X_RAW_002_RF'] - raw_df['COM_X_RAW']) * np.sin(-raw_df['COM_YAW_RAW']) + (raw_df['FT_Y_RAW_002_RF'] - raw_df['COM_Y_RAW']) * np.cos(-raw_df['COM_YAW_RAW'])
        raw_df['FT_Y_RAW_003_RH_body'] = (raw_df['FT_X_RAW_003_RH'] - raw_df['COM_X_RAW']) * np.sin(-raw_df['COM_YAW_RAW']) + (raw_df['FT_Y_RAW_003_RH'] - raw_df['COM_Y_RAW']) * np.cos(-raw_df['COM_YAW_RAW'])

        # sin(psi) + cos(psi)
        raw_df['COP_Y_body'] = ( raw_df['FT_Y_RAW_000_LF_body'] * raw_df['FT_FORCE_RAW_000_LF'] + raw_df['FT_Y_RAW_001_LH_body'] * raw_df['FT_FORCE_RAW_001_LH'] + raw_df['FT_Y_RAW_002_RF_body'] * raw_df['FT_FORCE_RAW_002_RF'] + raw_df['FT_Y_RAW_003_RH_body'] * raw_df['FT_FORCE_RAW_003_RH'] ) / (raw_df['FT_FORCE_RAW_000_LF'] + raw_df['FT_FORCE_RAW_001_LH'] + raw_df['FT_FORCE_RAW_002_RF'] + raw_df['FT_FORCE_RAW_003_RH']) 

        # Load additional DataFrames
        data_frames = {
            'NETWORK_OBS': pd.read_csv(data_path + 'obs.csv', header=None),
            'NETWORK_CN_IN': pd.read_csv(data_path + 'cn_in.csv', header=None),
            'NETWORK_HN_IN': pd.read_csv(data_path + 'hn_in.csv', header=None),
            'NETWORK_HN_OUT': pd.read_csv(data_path + 'hn_out.csv', header=None),
            'NETWORK_OBS_GRAD': pd.read_csv(data_path + 'obs_grad.csv', header=None),
            'NETWORK_CN_IN_GRAD': pd.read_csv(data_path + 'cn_in_grad.csv', header=None),
            'NETWORK_HN_IN_GRAD': pd.read_csv(data_path + 'hn_in_grad.csv', header=None),
            'NETWORK_HN_OUT_GRAD': pd.read_csv(data_path + 'hn_out_grad.csv', header=None)
        }

        data_frames['NETWORK_OBS_GRAD_TIMES_VALUE'] = np.tile(data_frames['NETWORK_OBS'].values, (1, 12)) * data_frames['NETWORK_OBS_GRAD']
        data_frames['NETWORK_HN_OUT_GRAD_TIMES_VALUE'] = np.tile(data_frames['NETWORK_HN_OUT'].values, (1, 12)) * data_frames['NETWORK_HN_OUT_GRAD']
        data_frames['NETWORK_CN_IN_GRAD_TIMES_VALUE'] = np.tile(data_frames['NETWORK_CN_IN'].values, (1, 12)) * data_frames['NETWORK_CN_IN_GRAD']
        data_frames['NETWORK_HN_IN_GRAD_TIMES_VALUE'] = np.tile(data_frames['NETWORK_HN_IN'].values, (1, 12)) * data_frames['NETWORK_HN_IN_GRAD']

        # Function to generate column names
        def generate_column_names(prefix, num_columns):
            return [f'{prefix}_{str(i).zfill(3)}' for i in range(num_columns)]

        # Concatenate all DataFrames column-wise with appropriate column names
        for prefix, df in data_frames.items():
            num_columns = df.shape[1]
            column_names = generate_column_names(prefix, num_columns)
            
            # Ensure the DataFrame to concatenate has the right column names
            df.columns = column_names
            
            # Concatenate the DataFrame column-wise
            raw_df = pd.concat([raw_df, df], axis=1)

        dfs_collection[idx] = raw_df
        segments_collections[idx] = precompute_segments(raw_df, contact_fields)

        # Define the global time window here
        global_time_window = (-1, 2.5)  # Replace 'start_time' and 'end_time' with actual values or references to cfg

        # Filter each DataFrame in dfs_collection according to the global_time_window
        for idx, df in dfs_collection.items():
            # Ensure TIME_RAW is within the global_time_window
            dfs_collection[idx] = df[(df['TIME_RAW'] >= global_time_window[0]) & (df['TIME_RAW'] <= global_time_window[1])]


        print('hi')

    all_plot_configurations = [
        {
            'plot_name': 'comparing_key_signals_across_models',
            'fields_and_nicknames': [
                ('TIME_RAW', 'OBS_RAW_001_v', 'v [m/s]'),
                ('TIME_RAW', 'ACT_RAW_009_RH_HAA', 'RH Hip Torque Command [Nm]'),
                ('TIME_RAW', 'OBS_RAW_021_dof_pos_angle_deg_10_RH_HAA', 'RH Hip Position [deg]'),
                ('TIME_RAW', 'ACT_RAW_006_RF_HAA', 'RF Hip Torque Command [Nm]'),
                ('TIME_RAW', 'OBS_RAW_018_dof_pos_angle_deg_07_RF_HAA', 'RF Hip Position [deg]'),
                # ('TIME_RAW', 'OBS_RAW_003_p_deg_s', r'$p$ [deg/s]'),
                # ('TIME_RAW', 'OBS_RAW_004_q_deg_s', r'$q$ [deg/s]'),
                ('TIME_RAW', 'OBS_RAW_006_phi_angle_deg', r'$\phi$ [deg]'),
                ('TIME_RAW', 'OBS_RAW_007_theta_angle_deg', r'$\theta$ [deg]'),
                ('TIME_RAW', 'COM_YAW_RAW', r'$\psi$ [deg]'),
            ],
            'plot_func': overlay_plot_signal,
            'fig_width': 6,
            'fig_height': 1.25,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_gaits_across_different_legs',
            'fields_and_nicknames': [
                ('TIME_RAW', 'LF', 'LF'),
                ('TIME_RAW', 'LH', 'LH'),
                ('TIME_RAW', 'RH', 'RH'),
                ('TIME_RAW', 'RF', 'RF'),
            ],
            'plot_func': overlay_plot_gait,
            'fig_width': 6,
            'fig_height': 1,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_joint_actuation_across_models',
            'fields_and_nicknames': [
                ('TIME_RAW', 'ACT_RAW_000_LF_HAA', 'ACT_LF_HAA'),
                ('TIME_RAW', 'ACT_RAW_001_LF_HFE', 'ACT_LF_HFE'),
                ('TIME_RAW', 'ACT_RAW_002_LF_KFE', 'ACT_LF_KFE'),
                ('TIME_RAW', 'ACT_RAW_003_LH_HAA', 'ACT_LH_HAA'),
                ('TIME_RAW', 'ACT_RAW_004_LH_HFE', 'ACT_LH_HFE'),
                ('TIME_RAW', 'ACT_RAW_005_LH_KFE', 'ACT_LH_KFE'),
                ('TIME_RAW', 'ACT_RAW_006_RF_HAA', 'ACT_RF_HAA'),
                ('TIME_RAW', 'ACT_RAW_007_RF_HFE', 'ACT_RF_HFE'),
                ('TIME_RAW', 'ACT_RAW_008_RF_KFE', 'ACT_RF_KFE'),
                ('TIME_RAW', 'ACT_RAW_009_RH_HAA', 'ACT_RH_HAA'),
                ('TIME_RAW', 'ACT_RAW_010_RH_HFE', 'ACT_RH_HFE'),
                ('TIME_RAW', 'ACT_RAW_011_RH_KFE', 'ACT_RH_KFE'),
            ],
            'plot_func': overlay_plot_signal,
            'fig_width': 6,
            'fig_height': 1.25,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_joint_pos_across_models',
            'fields_and_nicknames': [
                ('TIME_RAW', 'OBS_RAW_012_dof_pos_angle_deg_01_LF_HAA', 'POS_LF_HAA'),
                ('TIME_RAW', 'OBS_RAW_013_dof_pos_angle_deg_02_LF_HFE', 'POS_LF_HFE'),
                ('TIME_RAW', 'OBS_RAW_014_dof_pos_angle_deg_03_LF_KFE', 'POS_LF_KFE'),
                ('TIME_RAW', 'OBS_RAW_015_dof_pos_angle_deg_04_LH_HAA', 'POS_LH_HAA'),
                ('TIME_RAW', 'OBS_RAW_016_dof_pos_angle_deg_05_LH_HFE', 'POS_LH_HFE'),
                ('TIME_RAW', 'OBS_RAW_017_dof_pos_angle_deg_06_LH_KFE', 'POS_LH_KFE'),
                ('TIME_RAW', 'OBS_RAW_018_dof_pos_angle_deg_07_RF_HAA', 'POS_RF_HAA'),
                ('TIME_RAW', 'OBS_RAW_019_dof_pos_angle_deg_08_RF_HFE', 'POS_RF_HFE'),
                ('TIME_RAW', 'OBS_RAW_020_dof_pos_angle_deg_09_RF_KFE', 'POS_RF_KFE'),
                ('TIME_RAW', 'OBS_RAW_021_dof_pos_angle_deg_10_RH_HAA', 'POS_RH_HAA'),
                ('TIME_RAW', 'OBS_RAW_022_dof_pos_angle_deg_11_RH_HFE', 'POS_RH_HFE'),
                ('TIME_RAW', 'OBS_RAW_023_dof_pos_angle_deg_12_RH_KFE', 'POS_RH_KFE'),
            ],
            'plot_func': overlay_plot_signal,
            'fig_width': 6,
            'fig_height': 1.25,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_foot_placement_relative_to_com_across_diffent_legs',
            'fields_and_nicknames': [
                ('TIME_RAW', 'FT_Y_RAW_000_LF_body', 'FT_Y_LF'),
                ('TIME_RAW', 'FT_Y_RAW_001_LH_body', 'FT_Y_LH'),
                ('TIME_RAW', 'FT_Y_RAW_003_RH_body', 'FT_Y_RH'),
                ('TIME_RAW', 'FT_Y_RAW_002_RF_body', 'FT_Y_RF'),
                # ('TIME_RAW', 'COP_Y_body', 'COP_Y'),
            ],
            'plot_func': overlay_plot_gait,
            'fig_width': 6,
            'fig_height': 1.5,
            'additional_features': {
                'draw_perturb_shaded_box': True,
                'draw_centerline': True
            },
        },
        {
            'plot_name': 'comparing_joint_pos_across_models',
            'fields_and_nicknames': [
                ('TIME_RAW', 'FT_FORCE_RAW_000_LF', 'FT_FORCE_LF'),
                ('TIME_RAW', 'FT_FORCE_RAW_001_LH', 'FT_FORCE_LH'),
                ('TIME_RAW', 'FT_FORCE_RAW_003_RH', 'FT_FORCE_RH'),
                ('TIME_RAW', 'FT_FORCE_RAW_002_RF', 'FT_FORCE_RF'),
            ],
            'plot_func': overlay_plot_signal,
            'fig_width': 6,
            'fig_height': 2.5,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        # {
        #     'plot_name': 'comparing_first12_obs_gradients_rel_to_09_RH_HAA_actuation_across_models',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_OBS_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(9*188,9*188+12)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 3,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_09_RH_HAA_actuation_across_models_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_OBS_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(9*188,10*188)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_hn_out_gradients_rel_to_09_RH_HAA_actuation_across_models_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_HN_OUT_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(1152,1280)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_09_RH_HAA_actuation_across_models_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(1152,1280)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_hn_in_gradients_rel_to_09_RH_HAA_actuation_across_models_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_HN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(1152,1280)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_00_LF_HAA_actuation_across_models_plot',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_001', 'u'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_002', 'v'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_003', 'w'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_004', 'p'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_005', 'q'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_006', 'r'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_007', 'theta_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_008', 'phi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_009', 'psi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_010', 'u_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_011', 'v_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_012', 'r_star'),
        #     ],
        #     'plot_func': overlay_plot_signal,
        #     'fig_width': 12,
        #     'fig_height': 1,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_01_LF_HFE_actuation_across_models_plot',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_188', 'u'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_189', 'v'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_190', 'w'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_191', 'p'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_192', 'q'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_193', 'r'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_194', 'theta_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_195', 'phi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_196', 'psi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_197', 'u_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_198', 'v_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_199', 'r_star'),
        #     ],
        #     'plot_func': overlay_plot_signal,
        #     'fig_width': 12,
        #     'fig_height': 1,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_02_LF_KFE_actuation_across_models_plot',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_376', 'u'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_377', 'v'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_378', 'w'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_379', 'p'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_380', 'q'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_381', 'r'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_382', 'theta_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_383', 'phi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_384', 'psi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_385', 'u_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_386', 'v_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_387', 'r_star'),
        #     ],
        #     'plot_func': overlay_plot_signal,
        #     'fig_width': 12,
        #     'fig_height': 1,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        {
            'plot_name': 'comparing_obs_gradients_rel_to_03_LH_HAA_actuation_across_models_plot',
            'fields_and_nicknames': [
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_564', 'u'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_565', 'v'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_566', 'w'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_567', 'p'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_568', 'q'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_569', 'r'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_570', 'theta_proj'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_571', 'phi_proj'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_572', 'psi_proj'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_573', 'u_star'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_574', 'v_star'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_575', 'r_star'),
            ],
            'plot_func': overlay_plot_signal,
            'fig_width': 12,
            'fig_height': 1,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_04_LH_HFE_actuation_across_models_plot',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_752', 'u'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_753', 'v'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_754', 'w'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_755', 'p'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_756', 'q'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_757', 'r'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_758', 'theta_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_759', 'phi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_760', 'psi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_761', 'u_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_762', 'v_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_763', 'r_star'),
        #     ],
        #     'plot_func': overlay_plot_signal,
        #     'fig_width': 12,
        #     'fig_height': 1,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_05_LH_KFE_actuation_across_models_plot',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_940', 'u'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_941', 'v'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_942', 'w'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_943', 'p'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_944', 'q'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_945', 'r'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_946', 'theta_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_947', 'phi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_948', 'psi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_949', 'u_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_950', 'v_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_951', 'r_star'),
        #     ],
        #     'plot_func': overlay_plot_signal,
        #     'fig_width': 12,
        #     'fig_height': 1,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_06_RF_HAA_actuation_across_models_plot',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1128', 'u'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1129', 'v'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1130', 'w'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1131', 'p'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1132', 'q'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1133', 'r'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1134', 'theta_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1135', 'phi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1136', 'psi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1137', 'u_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1138', 'v_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1139', 'r_star'),
        #     ],
        #     'plot_func': overlay_plot_signal,
        #     'fig_width': 12,
        #     'fig_height': 1,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_07_RF_HFE_actuation_across_models_plot',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1316', 'u'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1317', 'v'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1318', 'w'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1319', 'p'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1320', 'q'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1321', 'r'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1322', 'theta_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1323', 'phi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1324', 'psi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1325', 'u_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1326', 'v_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1327', 'r_star'),
        #     ],
        #     'plot_func': overlay_plot_signal,
        #     'fig_width': 12,
        #     'fig_height': 1,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_08_RF_KFE_actuation_across_models_plot',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1504', 'u'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1505', 'v'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1506', 'w'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1507', 'p'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1508', 'q'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1509', 'r'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1510', 'theta_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1511', 'phi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1512', 'psi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1513', 'u_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1514', 'v_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1515', 'r_star'),
        #     ],
        #     'plot_func': overlay_plot_signal,
        #     'fig_width': 12,
        #     'fig_height': 1,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        {
            'plot_name': 'comparing_obs_gradients_rel_to_00_LF_HAA_actuation_across_models_plot',
            'fields_and_nicknames': [
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_000', 'u'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_001', 'v'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_002', 'w'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_003', 'p'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_004', 'q'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_005', 'r'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_006', 'theta_proj'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_007', 'phi_proj'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_008', 'psi_proj'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_009', 'u_star'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_010', 'v_star'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_011', 'r_star'),
            ],
            'plot_func': overlay_plot_signal,
            'fig_width': 12,
            'fig_height': 1,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_obs_gradients_rel_to_09_RH_HAA_actuation_across_models_plot',
            'fields_and_nicknames': [
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1692', 'u'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1693', 'v'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1694', 'w'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1695', 'p'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1696', 'q'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1697', 'r'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1698', 'theta_proj'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1699', 'phi_proj'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1700', 'psi_proj'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1701', 'u_star'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1702', 'v_star'),
                ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1703', 'r_star'),
            ],
            'plot_func': overlay_plot_signal,
            'fig_width': 12,
            'fig_height': 1,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_10_RH_HFE_actuation_across_models_plot',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1880', 'u'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1881', 'v'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1882', 'w'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1883', 'p'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1884', 'q'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1885', 'r'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1886', 'theta_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1887', 'phi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1888', 'psi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1889', 'u_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1890', 'v_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_1891', 'r_star'),
        #     ],
        #     'plot_func': overlay_plot_signal,
        #     'fig_width': 12,
        #     'fig_height': 1,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_obs_gradients_rel_to_11_RH_KFE_actuation_across_models_plot',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2068', 'u'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2069', 'v'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2070', 'w'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2071', 'p'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2072', 'q'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2073', 'r'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2074', 'theta_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2075', 'phi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2076', 'psi_proj'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2077', 'u_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2078', 'v_star'),
        #         ('TIME_RAW', 'NETWORK_OBS_GRAD_TIMES_VALUE_2079', 'r_star'),
        #     ],
        #     'plot_func': overlay_plot_signal,
        #     'fig_width': 12,
        #     'fig_height': 1,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_00_LF_HAA_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(0*128,1*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_01_LF_HFE_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(1*128,2*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_02_LF_KFE_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(2*128,3*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_03_LH_HAA_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(3*128,4*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_04_LH_HFE_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(4*128,5*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_05_LH_KFE_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(5*128,6*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_06_RF_HAA_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(6*128,7*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_07_RF_HFE_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(7*128,8*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_08_RF_KFE_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(8*128,9*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        {
            'plot_name': 'comparing_cn_in_values_rel_to_00_LF_HAA_actuation_across_models_plot_heatmap',
            'fields_and_nicknames': [
                ('TIME_RAW', f'NETWORK_CN_IN_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(0,128)  # Example for 5 dimensions
            ],
            'plot_func': overlay_plot_gradient_heatmaps,
            'fig_width': 12,
            'fig_height': 12,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_cn_in_gradients_rel_to_00_LF_HAA_actuation_across_models_plot_heatmap',
            'fields_and_nicknames': [
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(0*128,1*128)  # Example for 5 dimensions
            ],
            'plot_func': overlay_plot_gradient_heatmaps,
            'fig_width': 12,
            'fig_height': 12,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_cn_in_gradients_times_value_rel_to_00_LF_HAA_actuation_across_models_plot_heatmap',
            'fields_and_nicknames': [
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(0*128,1*128)  # Example for 5 dimensions
            ],
            'plot_func': overlay_plot_gradient_heatmaps,
            'fig_width': 12,
            'fig_height': 12,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_cn_in_values_rel_to_09_RH_HAA_actuation_across_models_plot_heatmap',
            'fields_and_nicknames': [
                ('TIME_RAW', f'NETWORK_CN_IN_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(0,128)  # Example for 5 dimensions
            ],
            'plot_func': overlay_plot_gradient_heatmaps,
            'fig_width': 12,
            'fig_height': 12,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_cn_in_gradients_rel_to_09_RH_HAA_actuation_across_models_plot_heatmap',
            'fields_and_nicknames': [
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(9*128,10*128)  # Example for 5 dimensions
            ],
            'plot_func': overlay_plot_gradient_heatmaps,
            'fig_width': 12,
            'fig_height': 12,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_cn_in_gradients_times_value_rel_to_09_RH_HAA_actuation_across_models_plot_heatmap',
            'fields_and_nicknames': [
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(9*128,10*128)  # Example for 5 dimensions
            ],
            'plot_func': overlay_plot_gradient_heatmaps,
            'fig_width': 12,
            'fig_height': 12,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_10_RH_HFE_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(10*128,11*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        # {
        #     'plot_name': 'comparing_cn_in_gradients_rel_to_11_RH_KFE_actuation_across_models_plot_heatmap',
        #     'fields_and_nicknames': [
        #         ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(i).zfill(3)}', f'Gradient Heatmap {i}') for i in range(11*128,12*128)  # Example for 5 dimensions
        #     ],
        #     'plot_func': overlay_plot_gradient_heatmaps,
        #     'fig_width': 12,
        #     'fig_height': 12,
        #     'additional_features': {
        #         'draw_perturb_shaded_box': True,
        #     },
        # },
        {
            'plot_name': 'comparing_key_cn_in_6_gradients_rel_to_all_actuation_across_models_plot',
            'fields_and_nicknames': [
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(0*128+6).zfill(3)}', 'Gradient-LF_HAA-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(1*128+6).zfill(3)}', 'Gradient-LF_HFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(2*128+6).zfill(3)}', 'Gradient-LF_KFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(3*128+6).zfill(3)}', 'Gradient-LH_HAA-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(4*128+6).zfill(3)}', 'Gradient-LH_HFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(5*128+6).zfill(3)}', 'Gradient-LH_KFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(6*128+6).zfill(3)}', 'Gradient-RF_HAA-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(7*128+6).zfill(3)}', 'Gradient-RF_HFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(8*128+6).zfill(3)}', 'Gradient-RF_KFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(9*128+6).zfill(3)}', 'Gradient-RH_HAA-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(10*128+6).zfill(3)}', 'Gradient-RH_HFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(11*128+6).zfill(3)}', 'Gradient-RH_KFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
            ],
            'plot_func': overlay_plot_signal,
            'fig_width': 12,
            'fig_height': 12,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_key_cn_in_gradients_rel_to_all_actuation_across_models_plot',
            'fields_and_nicknames': [
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(0*128+6).zfill(3)}', 'Gradient-LF_HAA Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(1*128+6).zfill(3)}', 'Gradient-LF_HFE Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(2*128+6).zfill(3)}', 'Gradient-LF_KFE Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(3*128+6).zfill(3)}', 'Gradient-LH_HAA Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(4*128+6).zfill(3)}', 'Gradient-LH_HFE Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(5*128+6).zfill(3)}', 'Gradient-LH_KFE Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(6*128+6).zfill(3)}', 'Gradient-RF_HAA Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(7*128+6).zfill(3)}', 'Gradient-RF_HFE Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(8*128+6).zfill(3)}', 'Gradient-RF_KFE Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(9*128+6).zfill(3)}', 'Gradient-RH_HAA Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(10*128+6).zfill(3)}', 'Gradient-RH_HFE Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_{str(11*128+6).zfill(3)}', 'Gradient-RH_KFE Recurrent Input Neuron 6'),  # Example for 5 dimensions
            ],
            'plot_func': overlay_plot_gait,
            'fig_width': 12,
            'fig_height': 12,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_key_cn_in_6_across_models_plot',
            'fields_and_nicknames': [
                ('TIME_RAW', f'NETWORK_CN_IN_{str(0*128+6).zfill(3)}', 'Recurrent Input Neuron 6'),  # Example for 5 dimensions
            ],
            'plot_func': overlay_plot_gait,
            'fig_width': 12,
            'fig_height': 12,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'plot_name': 'comparing_key_cn_in_6_gradients_times_val_rel_to_all_actuation_across_models_plot',
            'fields_and_nicknames': [
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(0*128+6).zfill(3)}', 'Gradient-LF_HAA-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(1*128+6).zfill(3)}', 'Gradient-LF_HFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(2*128+6).zfill(3)}', 'Gradient-LF_KFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(3*128+6).zfill(3)}', 'Gradient-LH_HAA-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(4*128+6).zfill(3)}', 'Gradient-LH_HFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(5*128+6).zfill(3)}', 'Gradient-LH_KFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(6*128+6).zfill(3)}', 'Gradient-RF_HAA-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(7*128+6).zfill(3)}', 'Gradient-RF_HFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(8*128+6).zfill(3)}', 'Gradient-RF_KFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(9*128+6).zfill(3)}', 'Gradient-RH_HAA-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(10*128+6).zfill(3)}', 'Gradient-RH_HFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
                ('TIME_RAW', f'NETWORK_CN_IN_GRAD_TIMES_VALUE_{str(11*128+6).zfill(3)}', 'Gradient-RH_KFE-times-value Recurrent Input Neuron 6'),  # Example for 5 dimensions
            ],
            'plot_func': overlay_plot_gait,
            'fig_width': 12,
            'fig_height': 12,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
    ]

    # Loop through each plot configuration
    for plot_config in all_plot_configurations:
        fields_and_nicknames = plot_config['fields_and_nicknames']
        plot_func = plot_config['plot_func']
        fig_width = plot_config['fig_width']
        fig_height = plot_config['fig_height']
        additional_features = plot_config.get('additional_features', {})  # Default to empty dict if not present

        # Create the directory if it doesn't exist
        os.makedirs(output_path, exist_ok=True)
        
        # Call the plot function with the unpacked arguments
        plot_path_name = output_path + plot_config['plot_name']
        plot_func(fields_and_nicknames, dfs_collection, segments_collections, colors, fig_width, fig_height, additional_features=additional_features, plot_name=plot_path_name)

    # Example usage
    plt.show()

    print('hi')

if __name__ == "__main__":
    run_analysis()