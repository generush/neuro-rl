import pandas as pd

from utils.data_processing import filter_by_column_keywords
from analysis.analyze_traj import analyze_traj
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

    # Filter DataFrame
    traj_df = filter_by_column_keywords(raw_df, cfg.dataset_names, 'RAW')

    # Compute pca for peturbed data
    traj_pc_df, pca_dict_perturbed = append_pc(traj_df, cfg.dataset_names, cfg.max_num_principal_components, cfg.normalization_type, 'PC', cfg.output_path)

    tf_path = 'data/processed/ANYMAL-1.0MASS-LSTM16-DISTTERR-01_U-0.4-1.0-7-50_UNPERTURBED/'
    traj_pc_df, pca_dict_unperturbed = append_pc(traj_pc_df, cfg.dataset_names, cfg.max_num_principal_components, cfg.normalization_type, 'UNPERTURBEDPC', cfg.output_path, tf_path)

    # Transform perturbed dataset to nominal unperturbed PCA space

    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot figures with speed colors and tangling colors
    ax.scatter(traj_pc_df['A_LSTM_HC_PC_000'], traj_pc_df['A_LSTM_HC_PC_001'], traj_pc_df['A_LSTM_HC_PC_002'], c=traj_pc_df['A_LSTM_HC_PC_002'], alpha=1, depthshade=True, rasterized=True)


    traj_pc_df.to_parquet(cfg.output_path + 'TRAJPERTURBED_UNPERTURBEDPC_DATA.parquet')
    traj_pc_df.to_csv(cfg.output_path + 'TRAJPERTURBED_UNPERTURBEDPC_CYCLE_DATA.csv')




    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot figures with speed colors and tangling colors
    ax.plot(traj_pc_df['A_LSTM_HC_PC_000'], traj_pc_df['A_LSTM_HC_PC_001'], traj_pc_df['A_LSTM_HC_PC_002'], alpha=1, rasterized=True)





    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot figures with speed colors and tangling colors
    ax.plot(traj_pc_df['A_LSTM_HC_UNPERTURBEDPC_000'], traj_pc_df['A_LSTM_HC_UNPERTURBEDPC_001'], traj_pc_df['A_LSTM_HC_UNPERTURBEDPC_002'], alpha=1, rasterized=True)






    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()

    # Filter columns: select only those starting with 'A_LSTM_HC_PC_'
    columns_to_plot = [col for col in traj_pc_df.columns if col.startswith('A_LSTM_HC_PC_')]

    # Loop through each filtered column and plot it
    for column in columns_to_plot:
        plt.plot(traj_pc_df.index, traj_pc_df[column], label=column)

    # Add legend to the plot to differentiate the lines
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

    # Add titles and labels (optional)
    plt.title('Overlay Plot of Specific Columns')
    plt.xlabel('Index')
    plt.ylabel('Value')

    # Show the plot
    plt.show()
    
    print('done')

if __name__ == "__main__":
    run_analysis()