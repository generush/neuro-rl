import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_processing import filter_by_column_keywords
from analysis.compute_avg_gait_cycle import compute_avg_gait_cycle
from analysis.append_pc import append_pc
from analysis.append_speed_axis import append_speed_axis
from analysis.compute_interpolation import compute_interpolation 
from analysis.plot_pc12_speed_axis import plot_pc12_speed_axis
from analysis.append_tangling import append_tangling
from analysis.compute_fixed_points import compute_fixed_points
from plotting.dashboard import run_dashboard

from cfg.loader import load_configuration

def run_analysis():
        
    cfg = load_configuration()

    dfs_collection = {}

    for idx, (data_path, model_path, output_path) in enumerate(zip(cfg.data_path, cfg.model_path, cfg.output_path)):

        print(data_path, model_path, output_path)

        # Load DataFrame
        raw_df = pd.read_parquet(data_path + 'RAW_DATA' + '.parquet')

        # Conver to deg
        raw_df['OBS_RAW_012_dof_pos_01'] = 180 / np.pi * (raw_df['OBS_RAW_012_dof_pos_01'])
        raw_df['OBS_RAW_013_dof_pos_02'] = 180 / np.pi * (raw_df['OBS_RAW_013_dof_pos_02'])
        raw_df['OBS_RAW_014_dof_pos_03'] = 180 / np.pi * (raw_df['OBS_RAW_014_dof_pos_03'])
        raw_df['OBS_RAW_015_dof_pos_04'] = 180 / np.pi * (raw_df['OBS_RAW_015_dof_pos_04'])
        raw_df['OBS_RAW_016_dof_pos_05'] = 180 / np.pi * (raw_df['OBS_RAW_016_dof_pos_05'])
        raw_df['OBS_RAW_017_dof_pos_06'] = 180 / np.pi * (raw_df['OBS_RAW_017_dof_pos_06'])
        raw_df['OBS_RAW_018_dof_pos_07'] = 180 / np.pi * (raw_df['OBS_RAW_018_dof_pos_07'])
        raw_df['OBS_RAW_019_dof_pos_08'] = 180 / np.pi * (raw_df['OBS_RAW_019_dof_pos_08'])
        raw_df['OBS_RAW_020_dof_pos_09'] = 180 / np.pi * (raw_df['OBS_RAW_020_dof_pos_09'])
        raw_df['OBS_RAW_021_dof_pos_10'] = 180 / np.pi * (raw_df['OBS_RAW_021_dof_pos_10'])
        raw_df['OBS_RAW_022_dof_pos_11'] = 180 / np.pi * (raw_df['OBS_RAW_022_dof_pos_11'])
        raw_df['OBS_RAW_023_dof_pos_12'] = 180 / np.pi * (raw_df['OBS_RAW_023_dof_pos_12'])

        # Compute OBS_RAW_007_theta
        raw_df['OBS_RAW_006_phi_proj'] = 180 / np.pi * np.arcsin(raw_df['OBS_RAW_007_theta_proj'])
        raw_df['OBS_RAW_007_theta_proj'] = 180 / np.pi * np.arcsin(raw_df['OBS_RAW_006_phi_proj'])
        
        # Find the index where PERTURB == 1 and align the TIME column based on this index
        perturb_index = raw_df[raw_df['PERTURB'] == 1].index[0]
        raw_df['TIME'] -= raw_df.loc[perturb_index, 'TIME']

        # Add columns for foot contacts
        raw_df['FT_CONTACT_LH'] = np.where(raw_df['FT_FORCE_RAW_001'] > 0, 1, np.nan)
        raw_df['FT_CONTACT_RH'] = np.where(raw_df['FT_FORCE_RAW_003'] > 0, 2, np.nan)
        raw_df['FT_CONTACT_RF'] = np.where(raw_df['FT_FORCE_RAW_002'] > 0, 3, np.nan)
        raw_df['FT_CONTACT_LF'] = np.where(raw_df['FT_FORCE_RAW_000'] > 0, 4, np.nan)

        # Add columns for foot contacts
        raw_df['FT_Y_LH'] = raw_df['FT_Y_RAW_001'].replace(0, np.nan)
        raw_df['FT_Y_RH'] = raw_df['FT_Y_RAW_003'].replace(0, np.nan)
        raw_df['FT_Y_RF'] = raw_df['FT_Y_RAW_002'].replace(0, np.nan)
        raw_df['FT_Y_LF'] = raw_df['FT_Y_RAW_000'].replace(0, np.nan)

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
        print('hi')
    import matplotlib.pyplot as plt

    def plot_data(dfs_collection, start_time=-1, end_time=1):

        for idx in dfs_collection:
            # Directly update the DataFrame in dfs_collection with the filtered data
            dfs_collection[idx] = dfs_collection[idx][(dfs_collection[idx]['TIME'] >= start_time) & (dfs_collection[idx]['TIME'] <= end_time)]
    
        colors = ['k', 'b', 'g', 'r']  # Define colors for plotting
        
        fields_to_plot1 = [
            'OBS_RAW_001_v',
            'ACT_RAW_009_RH_HAA',
            'OBS_RAW_021_dof_pos_10',
            'OBS_RAW_006_phi_proj',
        ]

        fields_to_plot2 = [
            'FT_CONTACT_LF',
            'FT_CONTACT_RF',
            'FT_CONTACT_LH',
            'FT_CONTACT_RH',
        ]

        fields_to_plot3 = [
            'ACT_RAW_000_LF_HAA',
            'ACT_RAW_001_LF_HFE',
            'ACT_RAW_002_LF_KFE',
            'ACT_RAW_003_LH_HAA',
            'ACT_RAW_004_LH_HFE',
            'ACT_RAW_005_LH_KFE',
            'ACT_RAW_006_RF_HAA',
            'ACT_RAW_007_RF_HFE',
            'ACT_RAW_008_RF_KFE',
            'ACT_RAW_009_RH_HAA',
            'ACT_RAW_010_RH_HFE',
            'ACT_RAW_011_RH_KFE'
        ]

        fields_to_plot4 = [
            'OBS_RAW_012_dof_pos_01',
            'OBS_RAW_013_dof_pos_02',
            'OBS_RAW_014_dof_pos_03',
            'OBS_RAW_015_dof_pos_04',
            'OBS_RAW_016_dof_pos_05',
            'OBS_RAW_017_dof_pos_06',
            'OBS_RAW_018_dof_pos_07',
            'OBS_RAW_019_dof_pos_08',
            'OBS_RAW_020_dof_pos_09',
            'OBS_RAW_021_dof_pos_10',
            'OBS_RAW_022_dof_pos_11',
            'OBS_RAW_023_dof_pos_12'
        ]

        fields_to_plot5 = [
            'FT_Y_LF',
            'FT_Y_RF',
            'FT_Y_LH',
            'FT_Y_RH',
            'COM_Y',
        ]

        field_nicknames1 = [
            'v [m/s]',
            'RH \nHip Torque \nCommand \n[Nm]',
            'RH \nHip Position \n[deg]',
            r'$\phi$ [deg/s]'
        ]

        field_nicknames2 = [
            'LF',
            'RF',
            'LH',
            'RH'
        ]
        
        field_nicknames3 = [
            'ACT_LF_HAA',
            'ACT_LF_HFE',
            'ACT_LF_KFE',
            'ACT_LH_HAA',
            'ACT_LH_HFE',
            'ACT_LH_KFE',
            'ACT_RF_HAA',
            'ACT_RF_HFE',
            'ACT_RF_KFE',
            'ACT_RH_HAA',
            'ACT_RH_HFE',
            'ACT_RH_KFE'
        ]
        
        field_nicknames4 = [
            'POS_LF_HAA',
            'POS_LF_HFE',
            'POS_LF_KFE',
            'POS_LH_HAA',
            'POS_LH_HFE',
            'POS_LH_KFE',
            'POS_RF_HAA',
            'POS_RF_HFE',
            'POS_RF_KFE',
            'POS_RH_HAA',
            'POS_RH_HFE',
            'POS_RH_KFE'
        ]

        field_nicknames5 = [
            'FT_Y_LF',
            'FT_Y_RF',
            'FT_Y_LH',
            'FT_Y_RH',
            'COM_Y',
        ]

        # Function to create plots for a given set of fields and nicknames
        def overlay_plot_signal(fields, nicknames, fig_width=10, fig_height=3):
            fig, axs = plt.subplots(len(fields), 1, figsize=(fig_width, fig_height * len(fields)), sharex=True)
            for idx, df in dfs_collection.items():
                for j, col in enumerate(fields):
                    if col in df.columns:
                        axs[j].plot(df['TIME'], df[col], label=f'DF{idx}', color=colors[idx % len(colors)])
                        axs[j].set_ylabel(nicknames[j], rotation=0, labelpad=50, fontsize='small')
                    else:
                        print(f"Column {col} not in DataFrame {idx}")
            for ax in axs:
                ax.axvspan(-0.02, -0.02+0.02, facecolor='lightgray', alpha=0.5)  # Translucent region
                ax.legend()
                ax.set_xlabel('Time [s]')
            plt.tight_layout()


        # Function to create plots for a given set of fields and nicknames
        def overlay_plot_gait(fields, nicknames, fig_width=10, fig_height=3):
            fig, axs = plt.subplots(len(fields), 1, figsize=(fig_width, fig_height * len(fields)), sharex=True)

            for idx, col in enumerate(fields):
                for j, df in dfs_collection.items():
                    if col in df.columns:
                        axs[j].plot(df['TIME'], df[col], label=nicknames[idx])
                        axs[j].set_ylabel(j, rotation=0, labelpad=50, fontsize='small')
                    else:
                        print(f"Column {col} not in DataFrame {idx}")
            for ax in axs:
                ax.axvspan(-0.02, -0.02+0.02, facecolor='lightgray', alpha=0.5)  # Translucent region
                ax.legend()
                ax.set_xlabel('Time [s]')
            plt.tight_layout()

        # Create the first plot
        overlay_plot_signal(fields_to_plot1, field_nicknames1, fig_width=8, fig_height=1.5)
        overlay_plot_gait(fields_to_plot2, field_nicknames2, fig_width=8, fig_height=1)
        overlay_plot_signal(fields_to_plot3, field_nicknames3, fig_width=8, fig_height=1.5)
        overlay_plot_signal(fields_to_plot4, field_nicknames4, fig_width=8, fig_height=1.5)
        overlay_plot_gait(fields_to_plot5, field_nicknames5, fig_width=8, fig_height=3)

    # At the end of your run_analysis function, call plot_data
    plot_data(dfs_collection, start_time=-1, end_time=1)

    plt.show()

    print('hi')




if __name__ == "__main__":
    run_analysis()

    print('done')



raw_df2 = pd.read_parquet('/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/data/raw/ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR/robustness_gradients_analysis/0/0.02/-12/last_AnymalTerrain_ep_4100_rew_20.68903.pth/RAW_DATA.parquet')
