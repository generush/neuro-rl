import pandas as pd

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

    # Load DataFrame
    raw_df = pd.read_parquet(cfg.data_path + 'RAW_DATA' + '.parquet')

    # Compute cycle-average and variance datasets from raw dataset
    avg_cycle_df, var_cycle_df = compute_avg_gait_cycle(raw_df)

    # Filter DataFrame
    avg_cycle_df = filter_by_column_keywords(avg_cycle_df, cfg.dataset_names, 'RAW')

    # Append pc data
    avg_cycle_df, pca_dict = append_pc(avg_cycle_df, cfg.dataset_names, cfg.max_num_principal_components, cfg.normalization_type, 'PC', cfg.output_path)
    
    # Append tangling data
    avg_cycle_df = append_tangling(avg_cycle_df, cfg.dataset_names, cfg.tangling_type)

    # Append speed axis
    avg_cycle_df = append_speed_axis(avg_cycle_df, cfg.dataset_names, cfg.max_num_principal_components, cfg.normalization_type, cfg.output_path)

    # Compute interpolated dataset from cycle-average dataset
    avg_cycle_interp_df = compute_interpolation(avg_cycle_df)

    plot_pc12_speed_axis(avg_cycle_interp_df, ['ACT', 'A_LSTM_HC'], cfg.output_path)

    # export processed data
    avg_cycle_df.to_parquet(cfg.output_path + 'AVG_CYCLE_DATA.parquet')
    avg_cycle_df.to_csv(cfg.output_path + 'AVG_CYCLE_DATA.csv')
    avg_cycle_interp_df.to_parquet(cfg.output_path + 'AVG_CYCLE_INTERP_DATA.parquet')
    avg_cycle_interp_df.to_csv(cfg.output_path + 'AVG_CYCLE_INTERP_DATA.csv')

    compute_fixed_points(cfg.model_path, cfg.output_path)
    
    print('done')

if __name__ == "__main__":
    run_analysis()