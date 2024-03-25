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

def overlay_plot_signal(fields_and_nicknames, dfs_collection, segments_collections, colors, fig_width, fig_height):

    # Determine the global min and max of the 'TIME_RAW' column across all DataFrames
    global_min_time = min(df['TIME_RAW'].min() for df in dfs_collection.values())
    global_max_time = max(df['TIME_RAW'].max() for df in dfs_collection.values())

    # Determine the number of subplots needed for the current group
    num_subplots = len(fields_and_nicknames)

    # Define the figure
    fig = plt.figure(figsize=(fig_width, num_subplots * fig_height), constrained_layout=True)

    for idx, (field, nickname) in enumerate(fields_and_nicknames):

        ax = fig.add_subplot(num_subplots, 1, idx + 1)  # Create a new subplot for each DataFrame

        for idx, df in dfs_collection.items():

            if field in df.columns:
                if any(key in field for key in segments_collections[0].keys()):
                    ax.plot(df['TIME_RAW'], df[field], label=f'DF{idx}', color=colors[idx], linewidth=0.5)

                    matching_keys = [key for key in segments_collections[idx].keys() if key in field]
                    if matching_keys:
                        for key in matching_keys:
                            segments_info = segments_collections[idx]
                            for start, end in zip(segments_info[key]['starts'], segments_info[key]['ends']):
                                ax.plot(df.loc[start:end, 'TIME_RAW'], df.loc[start:end, field], color=colors[idx], linewidth=2)
                else:
                    ax.plot(df['TIME_RAW'], df[field], label=f'DF{idx}', color=colors[idx], linewidth=1)
                
        # Find the start and end of the chunk where PERTURB_RAW=1
        if 'PERTURB_RAW' in df.columns:
            perturb_start = df[df['PERTURB_RAW'] == 1].min()['TIME_RAW']
            perturb_end = df[df['PERTURB_RAW'] == 1].max()['TIME_RAW']

            # Check if there is a perturb chunk in the DataFrame
            if pd.notna(perturb_start) and pd.notna(perturb_end):
                ax.axvspan(perturb_start, perturb_end, facecolor='lightgray', alpha=0.5)  # Translucent region

        ax.set_xlim([global_min_time, global_max_time])  # Set the same x-axis range for all subplots
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(nickname, rotation=0)
        ax.legend()

def overlay_plot_gait(fields_and_nicknames, dfs_collection, segments_collections, colors, fig_width, fig_height):
    
    # Determine the global min and max of the 'TIME_RAW' column across all DataFrames
    global_min_time = min(df['TIME_RAW'].min() for df in dfs_collection.values())
    global_max_time = max(df['TIME_RAW'].max() for df in dfs_collection.values())

    # Determine the number of subplots needed for the current group
    num_subplots = len(dfs_collection)

    # Define the figure
    fig = plt.figure(figsize=(fig_width, num_subplots * fig_height), constrained_layout=True)
    
    # Iterate over each DataFrame in the collection
    for idx, (df_key, df) in enumerate(dfs_collection.items()):

        ax = fig.add_subplot(num_subplots, 1, idx + 1)  # Create a new subplot for each DataFrame

        # Iterate over each signal field you want to plot for this DataFrame
        for field, nickname in fields_and_nicknames:
            if field in df.columns:
                if any(key in field for key in segments_collections[0].keys()):
                    ax.plot(df['TIME_RAW'], df[field], label=nickname, color=colors[idx], linewidth=0.5)

                # Check for matching segment keys
                matching_keys = [key for key in segments_collections[df_key].keys() if key in field]

                if matching_keys:
                    for key in matching_keys:
                        segments_info = segments_collections[df_key]
                        for start, end in zip(segments_info[key]['starts'], segments_info[key]['ends']):
                            ax.plot(df.loc[start:end, 'TIME_RAW'], df.loc[start:end, field], color=colors[idx], linewidth=2)
                else:
                    ax.plot(df['TIME_RAW'], df[field], label=nickname, color=colors[idx], linewidth=1)


        # Find the start and end of the chunk where PERTURB_RAW=1
        if 'PERTURB_RAW' in df.columns:
            perturb_start = df[df['PERTURB_RAW'] == 1].min()['TIME_RAW']
            perturb_end = df[df['PERTURB_RAW'] == 1].max()['TIME_RAW']

            # Check if there is a perturb chunk in the DataFrame
            if pd.notna(perturb_start) and pd.notna(perturb_end):
                ax.axvspan(perturb_start, perturb_end, facecolor='lightgray', alpha=0.5)  # Translucent region
                
        ax.set_xlim([global_min_time, global_max_time])  # Set the same x-axis range for all subplots
        ax.set_xlabel('Time [s]')
        ax.set_ylabel(f'DF{df_key}')
        ax.legend()
        
def run_analysis():
    cfg = load_configuration()
    dfs_collection = {}
    segments_collections = {}
    contact_fields = ['LF', 'LH', 'RH', 'RF']
    colors = ['k', 'b', 'g', 'r']

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
        raw_df['FT_Y_RAW_000_LF'] = raw_df['FT_Y_RAW_000']
        raw_df['FT_Y_RAW_001_LH'] = raw_df['FT_Y_RAW_001']
        raw_df['FT_Y_RAW_002_RF'] = raw_df['FT_Y_RAW_002']
        raw_df['FT_Y_RAW_003_RH'] = raw_df['FT_Y_RAW_003']
        
        # Create new columns with LF, LH, RH, RF in column name
        raw_df['FT_FORCE_RAW_000_LF'] = raw_df['FT_FORCE_RAW_000']
        raw_df['FT_FORCE_RAW_001_LH'] = raw_df['FT_FORCE_RAW_001']
        raw_df['FT_FORCE_RAW_003_RF'] = raw_df['FT_FORCE_RAW_002']
        raw_df['FT_FORCE_RAW_002_RH'] = raw_df['FT_FORCE_RAW_003']

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

        print('hi')

    # Plotting
    for idx, df in dfs_collection.items():
        fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)

        fields_and_nicknames1 = [
            ('OBS_RAW_001_v', 'v [m/s]'),
            ('ACT_RAW_009_RH_HAA', 'RH Hip Torque Command [Nm]'),
            ('OBS_RAW_018_dof_pos_angle_deg_07_RF_HAA', 'RH Hip Position [deg]'),
            ('OBS_RAW_006_phi_proj', r'$\phi$ [deg/s]')
        ]
        
        fields_and_nicknames2 = [
            ('LF', 'LF'),
            ('LH', 'LH'),
            ('RH', 'RH'),
            ('RF', 'RF'),
        ]

        fields_and_nicknames3 = [
            ('ACT_RAW_000_LF_HAA', 'ACT_LF_HAA'),
            ('ACT_RAW_001_LF_HFE', 'ACT_LF_HFE'),
            ('ACT_RAW_002_LF_KFE', 'ACT_LF_KFE'),
            ('ACT_RAW_003_LH_HAA', 'ACT_LH_HAA'),
            ('ACT_RAW_004_LH_HFE', 'ACT_LH_HFE'),
            ('ACT_RAW_005_LH_KFE', 'ACT_LH_KFE'),
            ('ACT_RAW_006_RF_HAA', 'ACT_RF_HAA'),
            ('ACT_RAW_007_RF_HFE', 'ACT_RF_HFE'),
            ('ACT_RAW_008_RF_KFE', 'ACT_RF_KFE'),
            ('ACT_RAW_009_RH_HAA', 'ACT_RH_HAA'),
            ('ACT_RAW_010_RH_HFE', 'ACT_RH_HFE'),
            ('ACT_RAW_011_RH_KFE', 'ACT_RH_KFE')
        ]

        fields_and_nicknames4 = [
            ('OBS_RAW_012_dof_pos_angle_deg_01_LF_HAA', 'POS_LF_HAA'),
            ('OBS_RAW_013_dof_pos_angle_deg_02_LF_HFE', 'POS_LF_HFE'),
            ('OBS_RAW_014_dof_pos_angle_deg_03_LF_KFE', 'POS_LF_KFE'),
            ('OBS_RAW_015_dof_pos_angle_deg_04_LH_HAA', 'POS_LH_HAA'),
            ('OBS_RAW_016_dof_pos_angle_deg_05_LH_HFE', 'POS_LH_HFE'),
            ('OBS_RAW_017_dof_pos_angle_deg_06_LH_KFE', 'POS_LH_KFE'),
            ('OBS_RAW_018_dof_pos_angle_deg_07_RF_HAA', 'POS_RF_HAA'),
            ('OBS_RAW_019_dof_pos_angle_deg_08_RF_HFE', 'POS_RF_HFE'),
            ('OBS_RAW_020_dof_pos_angle_deg_09_RF_KFE', 'POS_RF_KFE'),
            ('OBS_RAW_021_dof_pos_angle_deg_10_RH_HAA', 'POS_RH_HAA'),
            ('OBS_RAW_022_dof_pos_angle_deg_11_RH_HFE', 'POS_RH_HFE'),
            ('OBS_RAW_023_dof_pos_angle_deg_12_RH_KFE', 'POS_RH_KFE')
        ]

        fields_and_nicknames5 = [
            ('FT_Y_RAW_000_LF', 'FT_Y_LF'),
            ('FT_Y_RAW_001_LH', 'FT_Y_LH'),
            ('FT_Y_RAW_003_RH', 'FT_Y_RH'),
            ('FT_Y_RAW_002_RF', 'FT_Y_RF'),
            ('COM_Y_RAW', 'COM_Y'),
        ]


        fields_and_nicknames6 = [
            ('FT_FORCE_RAW_000_LF', 'FT_FORCE_LF'),
            ('FT_FORCE_RAW_001_LH', 'FT_FORCE_LH'),
            ('FT_FORCE_RAW_003_RH', 'FT_FORCE_RH'),
            ('FT_FORCE_RAW_002_RF', 'FT_FORCE_RF'),
        ]

    # Define fields and nicknames for all plots
    all_fields_and_nicknames = [
        (fields_and_nicknames1, overlay_plot_signal, 8, 1.5),
        (fields_and_nicknames2, overlay_plot_gait, 8, 1),
        (fields_and_nicknames3, overlay_plot_signal, 8, 1.5),
        (fields_and_nicknames4, overlay_plot_signal, 8, 1.5),
        (fields_and_nicknames5, overlay_plot_gait, 8, 3),
        (fields_and_nicknames6, overlay_plot_signal, 8, 3),
    ]

    # Loop through each group of fields and their associated plotting function
    for fields_and_nicknames, plot_func, fig_width, fig_height in all_fields_and_nicknames:
        plot_func(fields_and_nicknames, dfs_collection, segments_collections, colors, fig_width, fig_height)

        plt.tight_layout()
    plt.show()

    # # Determine the total number of subplots needed
    # total_subplots = sum(len(fields) for fields, plot_funcs, fig_width, fig_height in all_fields_and_nicknames)

    # # Create a figure with the appropriate number of subplots
    # fig, axs = plt.subplots(total_subplots, 1, figsize=(8, total_subplots * 1.5))
    # axs = axs.flatten()  # Flatten in case of a single column of subplots

    # # Counter for the current subplot index
    # subplot_idx = 0

    # # Loop through each group of fields and their associated plotting function
    # for fields_and_nicknames, plot_func, fig_width, fig_height in all_fields_and_nicknames:
    #     for field, nickname in fields_and_nicknames:
    #         ax = axs[subplot_idx]
    #         subplot_idx += 1
    #         plot_func(ax, dfs_collection, field, nickname, segments_collections, colors, fig_width, fig_height)

    # plt.tight_layout()
    # plt.show()

    print('hi')

if __name__ == "__main__":
    run_analysis()















# def run_analysis():
        
#     cfg = load_configuration()

#     dfs_collection = {}


#     for idx, (data_path, model_path, output_path) in enumerate(zip(cfg.data_path, cfg.model_path, cfg.output_path)):

#         print(data_path, model_path, output_path)

#         # Load DataFrame
#         raw_df = pd.read_parquet(data_path + 'RAW_DATA' + '.parquet')

#         # Replace 0's with NaN's
#         raw_df = raw_df.replace(0, np.nan)

#         # Conver to deg
#         raw_df['OBS_RAW_012_dof_pos_01'] = 180 / np.pi * (raw_df['OBS_RAW_012_dof_pos_01'])
#         raw_df['OBS_RAW_013_dof_pos_02'] = 180 / np.pi * (raw_df['OBS_RAW_013_dof_pos_02'])
#         raw_df['OBS_RAW_014_dof_pos_03'] = 180 / np.pi * (raw_df['OBS_RAW_014_dof_pos_03'])
#         raw_df['OBS_RAW_015_dof_pos_04'] = 180 / np.pi * (raw_df['OBS_RAW_015_dof_pos_04'])
#         raw_df['OBS_RAW_016_dof_pos_05'] = 180 / np.pi * (raw_df['OBS_RAW_016_dof_pos_05'])
#         raw_df['OBS_RAW_017_dof_pos_06'] = 180 / np.pi * (raw_df['OBS_RAW_017_dof_pos_06'])
#         raw_df['OBS_RAW_018_dof_pos_07'] = 180 / np.pi * (raw_df['OBS_RAW_018_dof_pos_07'])
#         raw_df['OBS_RAW_019_dof_pos_08'] = 180 / np.pi * (raw_df['OBS_RAW_019_dof_pos_08'])
#         raw_df['OBS_RAW_020_dof_pos_09'] = 180 / np.pi * (raw_df['OBS_RAW_020_dof_pos_09'])
#         raw_df['OBS_RAW_021_dof_pos_10'] = 180 / np.pi * (raw_df['OBS_RAW_021_dof_pos_10'])
#         raw_df['OBS_RAW_022_dof_pos_11'] = 180 / np.pi * (raw_df['OBS_RAW_022_dof_pos_11'])
#         raw_df['OBS_RAW_023_dof_pos_12'] = 180 / np.pi * (raw_df['OBS_RAW_023_dof_pos_12'])

#         # Compute OBS_RAW_007_theta
#         raw_df['OBS_RAW_006_phi_proj'] = 180 / np.pi * np.arcsin(raw_df['OBS_RAW_007_theta_proj'])
#         raw_df['OBS_RAW_007_theta_proj'] = 180 / np.pi * np.arcsin(raw_df['OBS_RAW_006_phi_proj'])
        
#         # Find the index where PERTURB == 1 and align the TIME column based on this index
#         perturb_index = raw_df[raw_df['PERTURB_RAW'] == 1].index[0]
#         raw_df['TIME_RAW'] -= raw_df.loc[perturb_index, 'TIME_RAW']

#         # Add columns for foot contacts
#         raw_df['FT_CONTACT_LF'] = np.where(raw_df['FT_FORCE_RAW_000'] > 0, 4, np.nan)
#         raw_df['FT_CONTACT_LH'] = np.where(raw_df['FT_FORCE_RAW_001'] > 0, 3, np.nan)
#         raw_df['FT_CONTACT_RH'] = np.where(raw_df['FT_FORCE_RAW_003'] > 0, 2, np.nan)
#         raw_df['FT_CONTACT_RF'] = np.where(raw_df['FT_FORCE_RAW_002'] > 0, 1, np.nan)

#         # Load additional DataFrames
#         data_frames = {
#             'NETWORK_OBS': pd.read_csv(data_path + 'obs.csv', header=None),
#             'NETWORK_CN_IN': pd.read_csv(data_path + 'cn_in.csv', header=None),
#             'NETWORK_HN_IN': pd.read_csv(data_path + 'hn_in.csv', header=None),
#             'NETWORK_HN_OUT': pd.read_csv(data_path + 'hn_out.csv', header=None),
#             'NETWORK_CN_IN_GRAD': pd.read_csv(data_path + 'cn_in_grad.csv', header=None),
#             'NETWORK_HN_IN_GRAD': pd.read_csv(data_path + 'hn_in_grad.csv', header=None),
#             'NETWORK_HN_OUT_GRAD': pd.read_csv(data_path + 'hn_out_grad.csv', header=None)
#         }

#         data_frames['NETWORK_HN_OUT_GRAD_TIMES_VALUE'] = data_frames['NETWORK_HN_OUT'] * data_frames['NETWORK_HN_OUT_GRAD']
#         data_frames['NETWORK_HN_IN_GRAD_TIMES_VALUE'] = data_frames['NETWORK_HN_IN'] * data_frames['NETWORK_HN_IN_GRAD']
#         data_frames['NETWORK_CN_IN_GRAD_TIMES_VALUE'] = data_frames['NETWORK_CN_IN'] * data_frames['NETWORK_CN_IN_GRAD']

#         # Function to generate column names
#         def generate_column_names(prefix, num_columns):
#             return [f'{prefix}_{str(i).zfill(3)}' for i in range(num_columns)]

#         # Concatenate all DataFrames column-wise with appropriate column names
#         for prefix, df in data_frames.items():
#             num_columns = df.shape[1]
#             column_names = generate_column_names(prefix, num_columns)
            
#             # Ensure the DataFrame to concatenate has the right column names
#             df.columns = column_names
            
#             # Concatenate the DataFrame column-wise
#             raw_df = pd.concat([raw_df, df], axis=1)

#         dfs_collection[idx] = raw_df
#         print('hi')

#     def plot_data(dfs_collection, start_time=-1, end_time=1):

#         for idx in dfs_collection:
#             # Directly update the DataFrame in dfs_collection with the filtered data
#             dfs_collection[idx] = dfs_collection[idx][(dfs_collection[idx]['TIME_RAW'] >= start_time) & (dfs_collection[idx]['TIME_RAW'] <= end_time)]
    
#         colors = ['k', 'b', 'g', 'r']  # Define colors for plotting

#         fields_and_nicknames1 = [
#             ('OBS_RAW_001_v', 'v [m/s]'),
#             ('ACT_RAW_009_RH_HAA', 'RH Hip Torque Command [Nm]'),
#             ('OBS_RAW_021_dof_pos_10', 'RH Hip Position [deg]'),
#             ('OBS_RAW_006_phi_proj', r'$\phi$ [deg/s]')
#         ]
        
#         fields_and_nicknames2 = [
#             ('FT_CONTACT_LF', 'LF'),
#             ('FT_CONTACT_LH', 'LH'),
#             ('FT_CONTACT_RH', 'RH'),
#             ('FT_CONTACT_RF', 'RF'),
#         ]

#         fields_and_nicknames3 = [
#             ('ACT_RAW_000_LF_HAA', 'ACT_LF_HAA'),
#             ('ACT_RAW_001_LF_HFE', 'ACT_LF_HFE'),
#             ('ACT_RAW_002_LF_KFE', 'ACT_LF_KFE'),
#             ('ACT_RAW_003_LH_HAA', 'ACT_LH_HAA'),
#             ('ACT_RAW_004_LH_HFE', 'ACT_LH_HFE'),
#             ('ACT_RAW_005_LH_KFE', 'ACT_LH_KFE'),
#             ('ACT_RAW_006_RF_HAA', 'ACT_RF_HAA'),
#             ('ACT_RAW_007_RF_HFE', 'ACT_RF_HFE'),
#             ('ACT_RAW_008_RF_KFE', 'ACT_RF_KFE'),
#             ('ACT_RAW_009_RH_HAA', 'ACT_RH_HAA'),
#             ('ACT_RAW_010_RH_HFE', 'ACT_RH_HFE'),
#             ('ACT_RAW_011_RH_KFE', 'ACT_RH_KFE')
#         ]

#         fields_and_nicknames4 = [
#             ('OBS_RAW_012_dof_pos_01', 'POS_LF_HAA'),
#             ('OBS_RAW_013_dof_pos_02', 'POS_LF_HFE'),
#             ('OBS_RAW_014_dof_pos_03', 'POS_LF_KFE'),
#             ('OBS_RAW_015_dof_pos_04', 'POS_LH_HAA'),
#             ('OBS_RAW_016_dof_pos_05', 'POS_LH_HFE'),
#             ('OBS_RAW_017_dof_pos_06', 'POS_LH_KFE'),
#             ('OBS_RAW_018_dof_pos_07', 'POS_RF_HAA'),
#             ('OBS_RAW_019_dof_pos_08', 'POS_RF_HFE'),
#             ('OBS_RAW_020_dof_pos_09', 'POS_RF_KFE'),
#             ('OBS_RAW_021_dof_pos_10', 'POS_RH_HAA'),
#             ('OBS_RAW_022_dof_pos_11', 'POS_RH_HFE'),
#             ('OBS_RAW_023_dof_pos_12', 'POS_RH_KFE')
#         ]

#         fields_and_nicknames5 = [
#             ('FT_Y_RAW_000', 'FT_Y_LF'),
#             ('FT_Y_RAW_001', 'FT_Y_LH'),
#             ('FT_Y_RAW_003', 'FT_Y_RH'),
#             ('FT_Y_RAW_002', 'FT_Y_RF'),
#             ('COM_Y_RAW', 'COM_Y'),
#         ]

#         # Function to create plots for a given set of fields and nicknames
#         def overlay_plot_signal(fields_and_nicknames, fig_width=10, fig_height=3):
#             fig, axs = plt.subplots(len(fields_and_nicknames), 1, figsize=(fig_width, fig_height * len(fields_and_nicknames)), sharex=True)
#             for field_idx, (field, nickname) in enumerate(fields_and_nicknames):
#                 for df_idx, df in dfs_collection.items():
#                     if field in df.columns:                                

#                         # Find the start and end indices of each continuous non-NaN segment
#                         lh_nan_mask = df['FT_CONTACT_RH'].isna()
#                         non_nan_starts_ends = df.index[~lh_nan_mask].values
#                         non_nan_segments = np.split(non_nan_starts_ends, np.where(np.diff(non_nan_starts_ends) != 1)[0]+1)

#                         # Plot each non-NaN segment individually
#                         for i, segment in enumerate(non_nan_segments):
#                             if len(segment) > 0:
#                                 label = nickname if df_idx == 0 and i == 0 else None  # Only label the first segment of the first dataframe
#                                 axs[field_idx].plot(df.loc[segment, 'TIME_RAW'], df.loc[segment, field], label=label, color=colors[df_idx % len(colors)], linewidth=1.5)

#                         # Find the start and end indices of each continuous NaN segment
#                         nan_starts_ends = df.index[lh_nan_mask].values
#                         nan_segments = np.split(nan_starts_ends, np.where(np.diff(nan_starts_ends) != 1)[0]+1)

#                         # Identify NaN segments and expand the window by one step on both ends
#                         for segment in nan_segments:
#                             if len(segment) > 0:
#                                 # Expanding the NaN window: one step earlier for start, one step later for end
#                                 start_index = max(segment[0] - 1, df.index[0])  # Ensure start index is not less than 0
#                                 end_index = min(segment[-1] + 1, df.index[-1])  # Ensure end index does not exceed the DataFrame length
                                
#                                 label = nickname if df_idx == 0 and i == 0 else None  # Only label the first segment of the first dataframe
#                                 axs[field_idx].plot(df.loc[start_index:end_index, 'TIME_RAW'], df.loc[start_index:end_index, field], label=label, color=colors[df_idx % len(colors)], linewidth=0.5)

#                         axs[field_idx].set_ylabel(nickname, rotation=0, labelpad=50, fontsize='small')
                       
#                     else:
#                         print(f"Column {field} not in DataFrame {idx}")
#                 for ax in axs:
#                     ax.axvspan(-0.02, -0.02+0.02, facecolor='lightgray', alpha=0.5)  # Translucent region
#                     ax.legend()
#                     ax.set_xlabel('Time [s]')
#                 plt.tight_layout()

#         # Function to create plots for a given set of fields and nicknames
#         def overlay_plot_gait(fields_and_nicknames, fig_width=10, fig_height=3):
#             fig, axs = plt.subplots(len(fields_and_nicknames), 1, figsize=(fig_width, fig_height * len(fields_and_nicknames)), sharex=True)
#             for field_idx, (field, nickname) in enumerate(fields_and_nicknames):
#                 for df_idx, df in dfs_collection.items():
#                     if field in df.columns:
#                         axs[df_idx].plot(df['TIME_RAW'], df[field], label=nickname, color=colors[df_idx % len(colors)])
#                         axs[df_idx].set_ylabel(f'DF{df_idx}', rotation=0, labelpad=50, fontsize='small')
#                     else:
#                         print(f"Column {field} not in DataFrame {idx}")
#                 for ax in axs:
#                     ax.axvspan(-0.02, -0.02+0.02, facecolor='lightgray', alpha=0.5)  # Translucent region
#                     ax.legend()
#                     ax.set_xlabel('Time [s]')
#                 plt.tight_layout()

#         # Create the first plot
#         overlay_plot_signal(fields_and_nicknames1, fig_width=8, fig_height=1.5)
#         overlay_plot_gait(fields_and_nicknames2, fig_width=8, fig_height=1)
#         overlay_plot_signal(fields_and_nicknames3, fig_width=8, fig_height=1.5)
#         overlay_plot_signal(fields_and_nicknames4, fig_width=8, fig_height=1.5)
#         overlay_plot_gait(fields_and_nicknames5, fig_width=8, fig_height=3)

#     # At the end of your run_analysis function, call plot_data
#     plot_data(dfs_collection, start_time=-1, end_time=1)

#     plt.show()

#     print('hi')




# if __name__ == "__main__":
#     run_analysis()

#     print('done')