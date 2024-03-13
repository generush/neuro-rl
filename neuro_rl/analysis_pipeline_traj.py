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

    tf_path = 'data/processed/old_2024_03_06/ANYMAL-1.0MASS-LSTM16-DISTTERR-99_U-0.4-1.0-7-50_UNPERTURBED/'
    traj_pc_df, pca_dict_unperturbed = append_pc(traj_pc_df, cfg.dataset_names, cfg.max_num_principal_components, cfg.normalization_type, 'UNPERTURBEDPC', cfg.output_path, tf_path)

    # Transform perturbed dataset to nominal unperturbed PCA space

    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot figures with speed colors and tangling colors
    # ax.scatter(traj_pc_df['A_LSTM_HC_PC_000'], traj_pc_df['A_LSTM_HC_PC_001'], traj_pc_df['A_LSTM_HC_PC_002'], c=traj_pc_df['A_LSTM_HC_PC_002'], alpha=1, depthshade=True, rasterized=True)
    ax.scatter(traj_pc_df['ACT_PC_000'], traj_pc_df['ACT_PC_001'], traj_pc_df['ACT_PC_002'], c=traj_pc_df['ACT_PC_002'], alpha=1, depthshade=True, rasterized=True)


    traj_pc_df.to_parquet(cfg.output_path + 'TRAJPERTURBED_UNPERTURBEDPC_DATA.parquet')
    traj_pc_df.to_csv(cfg.output_path + 'TRAJPERTURBED_UNPERTURBEDPC_CYCLE_DATA.csv')




    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot figures with speed colors and tangling colors
    # ax.plot(traj_pc_df['A_LSTM_HC_PC_000'], traj_pc_df['A_LSTM_HC_PC_001'], traj_pc_df['A_LSTM_HC_PC_002'], alpha=1, rasterized=True)
    ax.plot(traj_pc_df['ACT_PC_000'], traj_pc_df['ACT_PC_001'], traj_pc_df['ACT_PC_002'], alpha=1, rasterized=True)

    # Set labels and title
    ax.set_xlabel('PC 000')
    ax.set_ylabel('PC 001')
    ax.set_zlabel('PC 002')





    import torch
    import numpy as np
    import os
    import torch.nn as nn
    import pickle as pk

    # Check if the first file exists
    model_file = torch.load(os.path.join(cfg.model_path, 'nn/model.pth'))
    model_file = torch.load(os.path.join(cfg.model_path, 'nn/last_AnymalTerrain_ep_2200_rew_19.53241.pth'))
    model_file = torch.load(os.path.join(cfg.model_path, 'nn/last_AnymalTerrain_ep_200_rew_6.70734.pth'))
    model_file = torch.load(os.path.join(cfg.model_path, 'nn/last_AnymalTerrain_ep_24000_rew_20.797514.pth'))
    model_file = torch.load(os.path.join(cfg.model_path, '../ANYMAL-1.0MASS-LSTM16-BASELINE-99/nn/last_AnymalTerrain_ep_300_rew_15.173133.pth'))
    
    state_dict = {key.replace('a2c_network.a_rnn.rnn.', ''): value for key, value in model_file['model'].items() if key.startswith('a2c_network.a_rnn.rnn')}

    # get LSTM dimensions
    HIDDEN_SIZE = state_dict['weight_ih_l0'].size()[0] // 4
    INPUT_SIZE = state_dict['weight_ih_l0'].size()[1]
    N_LAYERS = max([int(key.split('_l')[-1]) for key in state_dict.keys() if key.startswith('weight_ih_l') or key.startswith('weight_hh_l')]) + 1

    # instantiate the LSTM and load weights
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    a_rnn = nn.LSTM(INPUT_SIZE, HIDDEN_SIZE, N_LAYERS).to(device)
    a_rnn.load_state_dict(state_dict)

    # load scaler and pca transforms
    scl = pk.load(open(cfg.output_path + 'A_LSTM_HC_SCL.pkl','rb'))
    pca = pk.load(open(cfg.output_path + 'A_LSTM_HC_PCA.pkl','rb'))

    import matplotlib.ticker as ticker

    # https://www.geeksforgeeks.org/matplotlib-pyplot-streamplot-in-python/#
    def plot_2d_streamplot(X, Y, U, V):

        # Create a figure and subplots
        plt.ion()
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # Ensure aspect ratio is equal to get a correct circle
        ax1.set_aspect('equal')


        # Plot quivers using the extracted components
        speed = np.sqrt(U**2 + V**2)
        lw = 5 * speed / speed.max()
        strm = ax1.streamplot(X, Y, U, V, density=2, linewidth=lw, color=speed/0.005, cmap ='plasma')
        plt.colorbar(strm.lines)

        # Create a ScalarFormatter and set the desired format
        formatter = ticker.ScalarFormatter(useMathText=True)
        formatter.set_scientific(False)
        formatter.set_powerlimits((-3, 4))  # Adjust the power limits if needed

        # Apply the formatter to the axis
        ax1.xaxis.set_major_formatter(formatter)
        ax1.yaxis.set_major_formatter(formatter)

        # Set labels and title
        ax1.set_xlabel('PC 1')
        ax1.set_ylabel('PC 2')

        # Show the legend
        # ax1.legend()

        return fig

    ### STREAMPLOT
    TRAJ_TIME_LENGTH = 100
    TRAJ_XY_DENSITY = 10

    input = torch.zeros((1, TRAJ_XY_DENSITY * TRAJ_XY_DENSITY, INPUT_SIZE), device=device,  dtype=torch.float32)

    # Create the X and Y meshgrid using torch.meshgrid
    X_RANGE = 30  # Adjust the value of X to set the range of the meshgrid
    x = torch.linspace(-X_RANGE, X_RANGE, TRAJ_XY_DENSITY)
    y = torch.linspace(-X_RANGE, X_RANGE, TRAJ_XY_DENSITY)
    YY, XX = torch.meshgrid(x, y)  # switch places of x and y

    # Reshape the X and Y meshgrid tensors into column vectors
    # meshgrid_tensor = torch.stack((XX.flatten(), XX.flatten()), dim=1)
    meshgrid_tensor = torch.stack((XX.flatten(), YY.flatten()), dim=1)

    # Expand the meshgrid tensor with zeros in the remaining columns
    zeros_tensor = torch.zeros(meshgrid_tensor.shape[0], 256 - 2)
    hc_zeroinput_t0_pc = torch.cat((meshgrid_tensor, zeros_tensor), dim=1).numpy()
    # hc_zeroinput_t0_pc[:,2:] = fps_pc[fp_idx,2:] # PC 3+ of fixed point (SLICE THROUGH PLANE PC1-PC2)

    hc = torch.tensor(scl.inverse_transform(pca.inverse_transform(hc_zeroinput_t0_pc)), dtype=torch.float32).unsqueeze(dim=0).to(device)

    # Extend hx_out in the first dimension
    hc_hist_zeroinput = torch.zeros((TRAJ_TIME_LENGTH,) + hc.shape[1:], dtype=hc.dtype)
    hc_hist_zeroinput[0,:,:] = hc

    for i in range(TRAJ_TIME_LENGTH - 1):

        # run step
        _, (hx, cx) = a_rnn(input, (hc[:,:,:HIDDEN_SIZE].contiguous(), hc[:,:,HIDDEN_SIZE:].contiguous()))
        hc = torch.cat((hx, cx), dim=2)

        hc_hist_zeroinput[i+1,:,:] = hc

    hc_zeroinput_tf_pc = pca.transform(scl.transform(torch.squeeze(hc_hist_zeroinput).reshape(-1, HIDDEN_SIZE * 2).detach().cpu().numpy())).reshape(hc_hist_zeroinput.shape)

    X = hc_zeroinput_t0_pc[:,0].reshape(TRAJ_XY_DENSITY, TRAJ_XY_DENSITY)
    Y = hc_zeroinput_t0_pc[:,1].reshape(TRAJ_XY_DENSITY, TRAJ_XY_DENSITY)
    U = (hc_zeroinput_tf_pc[1,:,0] - hc_zeroinput_tf_pc[0,:,0]).reshape(TRAJ_XY_DENSITY, TRAJ_XY_DENSITY)
    V = (hc_zeroinput_tf_pc[1,:,1] - hc_zeroinput_tf_pc[0,:,1]).reshape(TRAJ_XY_DENSITY, TRAJ_XY_DENSITY)
    fig = plot_2d_streamplot(X, Y, U, V)













    import matplotlib.pyplot as plt
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # plot figures with speed colors and tangling colors
    # ax.plot(traj_pc_df['A_LSTM_HC_UNPERTURBEDPC_000'], traj_pc_df['A_LSTM_HC_UNPERTURBEDPC_001'], traj_pc_df['A_LSTM_HC_UNPERTURBEDPC_002'], alpha=1, rasterized=True)
    ax.plot(traj_pc_df['ACT_UNPERTURBEDPC_000'], traj_pc_df['ACT_UNPERTURBEDPC_001'], traj_pc_df['ACT_UNPERTURBEDPC_002'], alpha=1, rasterized=True)









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