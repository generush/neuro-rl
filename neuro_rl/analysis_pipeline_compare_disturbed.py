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

        # Find the index where PERTURB == 1 and align the TIME column based on this index
        perturb_index = raw_df[raw_df['PERTURB'] == 1].index[0]
        raw_df['TIME'] -= raw_df.loc[perturb_index, 'TIME']

        # Add columns for foot contacts
        raw_df['FT_CONTACT_LF'] = np.where(raw_df['FT_FORCE_RAW_000'] > 0, 4, np.nan)
        raw_df['FT_CONTACT_LH'] = np.where(raw_df['FT_FORCE_RAW_001'] > 0, 1, np.nan)
        raw_df['FT_CONTACT_RF'] = np.where(raw_df['FT_FORCE_RAW_002'] > 0, 3, np.nan)
        raw_df['FT_CONTACT_RH'] = np.where(raw_df['FT_FORCE_RAW_003'] > 0, 2, np.nan)

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

    def plot_data(dfs_collection):
        # Create a figure and axes for subplots outside the loop
        fig, axs = plt.subplots(15, 1, figsize=(10, 12), sharex=True)
        colors = ['k', 'b', 'g', 'r']  # Define colors for plotting
        fields_to_plot = [
            'OBS_RAW_001_v',
            'ACT_RAW_009_RH_HAA',
            'OBS_RAW_021_dof_pos_10',
            'OBS_RAW_007_theta_proj',
            'ACT_RAW_000_LF_HAA',
            'ACT_RAW_001_LF_HFE',
            'ACT_RAW_002_LF_KFE',
            'ACT_RAW_003_LH_HAA',
            'ACT_RAW_004_LH_HFE',
            'ACT_RAW_005_LH_KFE',
            'ACT_RAW_006_RF_HAA',
            'ACT_RAW_007_RF_HFE',
            'ACT_RAW_008_RF_KFE',
            'ACT_RAW_010_RH_HFE',
            'ACT_RAW_011_RH_KFE'
        ]
        field_nicknames = [
            'v [m/s]',
            'RH \nHip Torque \nCommand \n[Nm]',
            'RH \nHip Position \n[deg]',
            r'$\phi$ [deg/s]',
            'ACT_RAW_000_LF_HAA',
            'ACT_RAW_001_LF_HFE',
            'ACT_RAW_002_LF_KFE',
            'ACT_RAW_003_LH_HAA',
            'ACT_RAW_004_LH_HFE',
            'ACT_RAW_005_LH_KFE',
            'ACT_RAW_006_RF_HAA',
            'ACT_RAW_007_RF_HFE',
            'ACT_RAW_008_RF_KFE',
            'ACT_RAW_010_RH_HFE',
            'ACT_RAW_011_RH_KFE'
        ]

        # Loop through the collection of DataFrames
        for idx, df in dfs_collection.items():
            # Plot specific fields on the same set of axes
            for j, col in enumerate(fields_to_plot):
                if col in df.columns:
                    axs[j].plot(df['TIME'], df[col], label=f'DF{idx}', color=colors[idx % len(colors)])
                    axs[j].set_ylabel(field_nicknames[j], rotation=0, labelpad=50)
                else:
                    print(f"Column {col} not in DataFrame {idx}")

        # Add light gray translucent region to each subplot (adjust times as needed)
        translucent_time_range = (0.0, 0.08)
        for ax in axs:
            ax.axvspan(translucent_time_range[0], translucent_time_range[1], facecolor='lightgray', alpha=0.5)

        # Customize legend and labels for each subplot
        for ax in axs:
            ax.legend()
            # ax.set_xlim(-0.8, 1.2)
            ax.set_xlabel('Time')

        # Save or show the figure
        plt.tight_layout()

    # Call plot_data with your dfs_collection after it's been populated
    # plot_data(dfs_collection)


    # At the end of your run_analysis function, call plot_data
    plot_data(dfs_collection)

    plt.show()

    print('hi')




if __name__ == "__main__":
    run_analysis()

    print('done')