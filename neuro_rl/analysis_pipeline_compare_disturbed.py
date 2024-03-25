import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def overlay_plot_signal(fields_and_nicknames, dfs_collection, segments_collections, colors, fig_width, fig_height, additional_features=None):

    # Determine the global min and max of the 'TIME_RAW' column across all DataFrames
    global_min_time = min(df['TIME_RAW'].min() for df in dfs_collection.values())
    global_max_time = max(df['TIME_RAW'].max() for df in dfs_collection.values())

    # Determine the number of subplots needed for the current group
    num_subplots = len(fields_and_nicknames)

    # Define the figure
    fig = plt.figure(figsize=(fig_width, num_subplots * fig_height), constrained_layout=True)

    for field_idx, (x_field, y_field, nickname) in enumerate(fields_and_nicknames):

        ax = fig.add_subplot(num_subplots, 1, field_idx + 1)  # Create a new subplot for each DataFrame

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

        ax.set_xlim([global_min_time, global_max_time])  # Set the same x-axis range for all subplots
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(nickname, rotation=0)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

def overlay_plot_gait(fields_and_nicknames, dfs_collection, segments_collections, colors, fig_width, fig_height, additional_features=None):
    
    # Determine the global min and max of the 'TIME_RAW' column across all DataFrames
    global_min_time = min(df['TIME_RAW'].min() for df in dfs_collection.values())
    global_max_time = max(df['TIME_RAW'].max() for df in dfs_collection.values())

    # Determine the number of subplots needed for the current group
    num_subplots = len(dfs_collection)

    # Define the figure
    fig = plt.figure(figsize=(fig_width, num_subplots * fig_height), constrained_layout=True)
    
    # Iterate over each DataFrame in the collection
    for df_idx, (df_key, df) in enumerate(dfs_collection.items()):

        ax = fig.add_subplot(num_subplots, 1, df_idx + 1)  # Create a new subplot for each DataFrame

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


                
        if x_field == 'TIME_RAW':
            ax.set_xlim([global_min_time, global_max_time])  # Set the same x-axis range for all subplots
            ax.set_xlabel('Time [s]')
        else:
            ax.set_xlabel(x_field)
            # Set the aspect ratio to be equal
            ax.set_aspect('equal', adjustable='box')
            
        ax.set_ylabel(f'DF{df_key}')
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
def run_analysis():
    cfg = load_configuration()

    dfs_collection = {}
    segments_collections = {}
    contact_fields = ['LF', 'LH', 'RH', 'RF']
    colors = ['k', 'r', 'b', 'g', 'm']

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

        # Compute OBS_RAW_007_theta
        raw_df['OBS_RAW_006_phi_angle_deg'] = 180 / np.pi * np.arcsin(raw_df['OBS_RAW_007_theta_proj'])
        raw_df['OBS_RAW_007_theta_angle_deg'] = 180 / np.pi * np.arcsin(raw_df['OBS_RAW_006_phi_proj'])
        
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
            'NETWORK_CN_IN_GRAD': pd.read_csv(data_path + 'cn_in_grad.csv', header=None),
            'NETWORK_HN_IN_GRAD': pd.read_csv(data_path + 'hn_in_grad.csv', header=None),
            'NETWORK_HN_OUT_GRAD': pd.read_csv(data_path + 'hn_out_grad.csv', header=None)
        }

        data_frames['NETWORK_HN_OUT_GRAD_TIMES_VALUE'] = data_frames['NETWORK_HN_OUT'] * data_frames['NETWORK_HN_OUT_GRAD']
        data_frames['NETWORK_HN_IN_GRAD_TIMES_VALUE'] = data_frames['NETWORK_HN_IN'] * data_frames['NETWORK_HN_IN_GRAD']
        data_frames['NETWORK_CN_IN_GRAD_TIMES_VALUE'] = data_frames['NETWORK_CN_IN'] * data_frames['NETWORK_CN_IN_GRAD']

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
        global_time_window = (-1, 1.5)  # Replace 'start_time' and 'end_time' with actual values or references to cfg

        # Filter each DataFrame in dfs_collection according to the global_time_window
        for idx, df in dfs_collection.items():
            # Ensure TIME_RAW is within the global_time_window
            dfs_collection[idx] = df[(df['TIME_RAW'] >= global_time_window[0]) & (df['TIME_RAW'] <= global_time_window[1])]


        print('hi')

    all_plot_configurations = [
        {
            'fields_and_nicknames': [
                ('TIME_RAW', 'OBS_RAW_001_v', 'v [m/s]'),
                ('TIME_RAW', 'ACT_RAW_009_RH_HAA', 'RH Hip Torque Command [Nm]'),
                ('TIME_RAW', 'OBS_RAW_018_dof_pos_angle_deg_07_RF_HAA', 'RH Hip Position [deg]'),
                ('TIME_RAW', 'OBS_RAW_007_theta_angle_deg', r'$\phi$ [deg]')
            ],
            'plot_func': overlay_plot_signal,
            'fig_width': 6,
            'fig_height': 1.5,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'fields_and_nicknames': [
                ('TIME_RAW', 'LF', 'LF'),
                ('TIME_RAW', 'LH', 'LH'),
                ('TIME_RAW', 'RH', 'RH'),
                ('TIME_RAW', 'RF', 'RF'),
            ],
            'plot_func': overlay_plot_gait,
            'fig_width': 6,
            'fig_height': 1.25,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
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
            'fig_height': 1.5,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
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
            'fig_height': 1.5,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        },
        {
            'fields_and_nicknames': [
                ('TIME_RAW', 'FT_Y_RAW_000_LF_body', 'FT_Y_LF'),
                ('TIME_RAW', 'FT_Y_RAW_001_LH_body', 'FT_Y_LH'),
                ('TIME_RAW', 'FT_Y_RAW_003_RH_body', 'FT_Y_RH'),
                ('TIME_RAW', 'FT_Y_RAW_002_RF_body', 'FT_Y_RF'),
                # ('TIME_RAW', 'COP_Y_body', 'COP_Y'),
            ],
            'plot_func': overlay_plot_gait,
            'fig_width': 6,
            'fig_height': 2,
            'additional_features': {
                'draw_perturb_shaded_box': True,
                'draw_centerline': True
            },
        },
        {
            'fields_and_nicknames': [
                ('TIME_RAW', 'FT_FORCE_RAW_000_LF', 'FT_FORCE_LF'),
                ('TIME_RAW', 'FT_FORCE_RAW_001_LH', 'FT_FORCE_LH'),
                ('TIME_RAW', 'FT_FORCE_RAW_003_RH', 'FT_FORCE_RH'),
                ('TIME_RAW', 'FT_FORCE_RAW_002_RF', 'FT_FORCE_RF'),
            ],
            'plot_func': overlay_plot_signal,
            'fig_width': 6,
            'fig_height': 3,
            'additional_features': {
                'draw_perturb_shaded_box': True,
            },
        }
    ]

    # Loop through each plot configuration
    for plot_config in all_plot_configurations:
        fields_and_nicknames = plot_config['fields_and_nicknames']
        plot_func = plot_config['plot_func']
        fig_width = plot_config['fig_width']
        fig_height = plot_config['fig_height']
        additional_features = plot_config.get('additional_features', {})  # Default to empty dict if not present

        # Call the plot function with the unpacked arguments
        plot_func(fields_and_nicknames, dfs_collection, segments_collections, colors, fig_width, fig_height, additional_features=additional_features)
        plt.tight_layout()

    plt.show()

    print('hi')

if __name__ == "__main__":
    run_analysis()