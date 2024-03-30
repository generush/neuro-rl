import pandas as pd

from utils.data_processing import filter_by_column_keywords
from analysis.compute_avg_gait_cycle import compute_avg_gait_cycle
from analysis.append_pc import append_pc
from analysis.append_speed_axis import append_speed_axis
from analysis.compute_interpolation import compute_interpolation
from analysis.plot_pc import plot_pc123
from analysis.append_tangling import append_tangling
from analysis.compute_fixed_points import compute_fixed_points
from plotting.dashboard import run_dashboard

from cfg.loader import load_configuration

def run_analysis():
        
    cfg = load_configuration()

    all_dfs = []

    for i, data_path in enumerate(cfg.data_path):
        df = pd.read_parquet(data_path + 'RAW_DATA' + '.parquet')

        # Find the time when PERTURB_RAW first becomes 1
        perturb_start_time = df[df['PERTURB_RAW'] == 1]['TIME_RAW'].min()

        # Define your time window from the perturbation start time
        # For example, to take a window of 0.3 time units after perturbation starts
        start_window = perturb_start_time
        end_window = perturb_start_time + 0.55

        # Filter the DataFrame based on the time window
        df_filtered = df[(df['TIME_RAW'] >= start_window) & (df['TIME_RAW'] <= end_window)]
        
        # Add a 'df_id' column to the front of the DataFrame
        df_filtered.insert(0, 'RUN_ID', i)  # 'i' is the unique identifier for each DataFrame
        
        # Add the filtered DataFrame to our list
        all_dfs.append(df_filtered)

    # Concatenate all DataFrames in the list, aligning them by columns
    combined_df = pd.concat(all_dfs, ignore_index=True)
    
    # Append pc data
    filt_pc_df, pca_dict = append_pc(combined_df, cfg.dataset_names, cfg.max_num_principal_components, cfg.normalization_type, 'PC', cfg.output_path)
    
    # Compute interpolated dataset from cycle-average dataset
    filt_pc_interp_df = compute_interpolation(filt_pc_df)

    plot_pc123(filt_pc_interp_df, ['OBS', 'ACT', 'A_LSTM_HC',  'A_LSTM_CX',  'A_LSTM_HX'], 'TIME_RAW', cfg.output_path)

    # export processed data
    filt_pc_df.to_parquet(cfg.output_path + 'FILT_PC_DATA.parquet')
    filt_pc_df.to_csv(cfg.output_path + 'FILT_PC_DATA.csv')
    filt_pc_interp_df.to_csv(cfg.output_path + 'FILT_PC_INTERP_DATA.csv')

    # compute_fixed_points(cfg.model_path, cfg.output_path)
    
    print('done')

if __name__ == "__main__":
    run_analysis()