# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from typing import List

import numpy as np
import pandas as pd

from utils.data_processing import process_data
from plotting.generation import generate_dropdown, generate_graph
from plotting.plot import plot_scatter3_ti_tf
from embeddings.embeddings import Data, Embeddings, MultiDimensionalScalingEmbedding, PCAEmbedding, MDSEmbedding, ISOMAPEmbedding,LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding

import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

def compute_pca(df_raw, n_components, columns):


    # # get column names that contain the string "RAW"
    # cols_to_normalize = [col for col in df_raw.columns if 'RAW' in col]

    # # normalize only the columns that contain the string "norm", avoiding normalization if max = min
    # col_min = df_raw[cols_to_normalize].min()
    # col_max = df_raw[cols_to_normalize].max()
    # col_range = (col_max - col_min).compute() + 5
    # col_range[col_range == 0] = 1 # avoid division by zero
    # normalized_df = df_raw[cols_to_normalize] / col_range

    # # concatenate the normalized dataframe with the original dataframe along the columns axis
    # df_raw = pd.concat([df_raw.drop(cols_to_normalize, axis=1).compute(), normalized_df.compute()], axis=1)


    # create PCA object
    pca = sklearn.decomposition.PCA(n_components=n_components)

    # fit pca transform
    data_pc = pca.fit_transform(df_raw)

    # create DataFrame
    df_pc = pd.DataFrame(data_pc)

    # name DataFrame columns
    df_pc.columns = columns

    return pca, df_pc

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk

def export_pca(pca: sklearn.decomposition.PCA, path: str):
    pk.dump(pca, open(path,"wb"))

def import_pca(path: str):
    return pk.load(open(path,'rb'))

# define the objective function to minimize
def objective(d, x, v):
    return np.linalg.norm(np.matmul(x, d) - v)

# define the constraint function that enforces ||d|| = 1
def constraint(d):
    return np.linalg.norm(d) - 1

def analyze_pca_speed_axis(path: str, data_names: List[str], file_suffix: str = ''):

    N_COMPONENTS = 12

    # load DataFrame
    df = process_data(path + 'RAW_DATA' + file_suffix + '.csv')

    dt = df['TIME'][1].compute().to_numpy() - df['TIME'][0].compute().to_numpy()

    for idx, data_type in enumerate(data_names):

        # select data for PCA analysis (only raw data)
        filt_data = df.loc[:,df.columns.str.contains(data_type + '_RAW')].compute()

        spd_cmd = df.loc[:,'OBS_RAW_009_u_star'].compute()
        spd_act = df.loc[:,'OBS_RAW_000_u'].compute()

        # get indices for jth condition
        idx = df[(df['OBS_RAW_010_v_star'] == 0) & (df['OBS_RAW_011_r_star'] == 0)].index.compute()

        df_neuron = filt_data.loc[idx].reset_index(drop=True)
        df_speed_cmd = spd_cmd.loc[idx].reset_index(drop=True)
        df_speed_act = spd_act.loc[idx].reset_index(drop=True)

        # get number of dimensions of DataFrame
        N_DIMENSIONS = len(df_neuron.columns)

        # create column name
        COLUMNS = np.char.mod(data_type + 'SPEED_U_PC_%03d', np.arange(N_COMPONENTS))

        if N_DIMENSIONS > 0:

            # computa pca
            pca, pc_df = compute_pca(df_neuron, N_COMPONENTS, COLUMNS)

            # export PCA object
            export_pca(pca, path + data_type +'_SPEED_U_PCA' + '.pkl')

            # export DataFrame
            pc_df.to_csv(path + data_type + '_SPEED_U_' + 'PC_DATA' + file_suffix + '.csv')
        
        x_s = pc_df.iloc[:,2:]
        
        v_bar = df_speed_cmd.unique()

        NUM_SPEEDS = len(v_bar)

        x_bar = np.zeros((NUM_SPEEDS, N_COMPONENTS - 2), dtype=float)
        # v_bar_meas = np.zeros(NUM_SPEEDS, dtype=float)

        for i in range(NUM_SPEEDS):
           x_bar[i,:] = x_s[df_speed_cmd==v_bar[i]].mean().to_numpy()
        #    v_bar_meas[i] = df_speed_act[df_speed_cmd==v_bar[i]].mean()


        from scipy.optimize import minimize

        # set the initial guess for d
        d0 = np.ones((x_s.shape[1], 1)) / x_s.shape[1]

        # set up the optimization problem
        problem = {
            'fun': objective,
            'x0': d0,
            'args': (x_bar / np.linalg.norm(x_bar), v_bar - v_bar.mean()), # mean-centered mean velocities
            'constraints': {'type': 'eq', 'fun': constraint}
        }

        # solve the optimization problem
        result = minimize(**problem)

        # extract the optimal value of d
        d_opt = result.x

        speed_axis = np.matmul ( pca.components_[2:].transpose(), d_opt )
        pc_speed_tf = np.column_stack((
            pca.components_[0],
            pca.components_[1],
            speed_axis
        ))

        pc_speed_df = pd.DataFrame(np.matmul ( df_neuron.to_numpy(), pc_speed_tf) )




        import matplotlib.pyplot as plt
        from scipy.interpolate import CubicSpline
        from mpl_toolkits.mplot3d.art3d import Line3DCollection
        import matplotlib.colors as mcolors
        import matplotlib.cm as cm

        # PLOT THE PRETTY INTERPOLATED TRAJECTORIES
        N_INTERP_TIMES = 1000
        xx = np.zeros((N_INTERP_TIMES, NUM_SPEEDS), dtype=float)
        yy = np.zeros((N_INTERP_TIMES, NUM_SPEEDS), dtype=float)
        zz = np.zeros((N_INTERP_TIMES, NUM_SPEEDS), dtype=float)
        ss = np.zeros((N_INTERP_TIMES, NUM_SPEEDS), dtype=float)


        import matplotlib.pyplot as plt
        from scipy.interpolate import CubicSpline
        import matplotlib.cm as cm

        c = df_speed_cmd.to_numpy()

        # Plot the original trajectory and the interpolated trajectory
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        color_vals = []  # Store the original interp_s_vals for color bar

        for idx, v in enumerate(v_bar):
            x = pc_df.iloc[:, 0][c == v].to_numpy()
            y = pc_df.iloc[:, 1][c == v].to_numpy()
            z = pc_df.iloc[:, 2][c == v].to_numpy()
            s = df_speed_cmd[c == v].to_numpy()

            # get the length of the trajectory
            n = len(x)

            # Create a periodic sequence for interpolation
            time = np.arange(0, n + 1) * dt  # Time points
            time_periodic = np.arange(0, len(x) + 1) * dt  # Time points

            # append first point to end to close the trajectory (no gaps in interpolation)
            x = np.append(x, x[0])
            y = np.append(y, y[0])
            z = np.append(z, z[0])
            s = np.append(s, s[0])

            # Perform periodic cubic spline interpolation for each dimension
            interp_x = CubicSpline(time_periodic, x, bc_type='periodic')
            interp_y = CubicSpline(time_periodic, y, bc_type='periodic')
            interp_z = CubicSpline(time_periodic, z, bc_type='periodic')
            interp_s = CubicSpline(time_periodic, s, bc_type='periodic')

            # Create a finer grid for interpolation
            fine_time = np.linspace(time[0], time[-1], num=1000)  # 1 ms

            # Evaluate the interpolating functions
            interp_x_vals = interp_x(fine_time)
            interp_y_vals = interp_y(fine_time)
            interp_z_vals = interp_z(fine_time)
            interp_s_vals = interp_s(fine_time)
            
            xx[:, idx] = interp_x_vals
            yy[:, idx] = interp_y_vals
            zz[:, idx] = interp_z_vals
            ss[:, idx] = interp_s_vals

        # Plot the interpolated trajectory as scatter with color corresponding to interp_s
        scatter1 = ax.scatter(xx, yy, zz, c=ss, s=2*np.ones_like(xx), cmap='bwr', alpha=1)

        # Add labels and a legend
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('Speed Axis')
        ax.set_title(data_type + " interpolation")

        # Add a 2D projection (x, y)
        scatter2 = ax.scatter(xx.flatten(), yy.flatten(), 1.5 * zz.min()*np.ones_like(zz), c='grey', s=1*np.ones_like(xx), alpha=1)

        # # Plot the original data points as scatter
        # scatter3 = ax.scatter(x, y, z, c=s, s=50*np.ones_like(x), alpha=1)

        # Define normalization for colorbars
        norm = mcolors.Normalize(vmin=ss.min(), vmax=ss.max())  # Adjust vmin and vmax according to your data

        # Create colorbars for each scatter plot
        cbar1 = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='bwr'), ax=ax)
        cbar1.set_label('Speed')

        plt.show()




        # import matplotlib.pyplot as plt
        # # create a 3D scatter plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # scatter = ax.scatter(pc_speed_df[0], pc_speed_df[1], pc_speed_df[2], c=df_speed_cmd)

        # # set axis labels and title
        # ax.set_xlabel('PC1')
        # ax.set_ylabel('PC2')
        # ax.set_zlabel('Speed Axis')
        # colorbar = plt.colorbar(scatter)
        # colorbar.set_label('forward velocity (m/s)')
        # # show the plot
        # plt.show()
        
        # # create a 3D scatter plot
        # fig2 = plt.figure()
        # ax2 = fig2.add_subplot(111, projection='3d')
        # scatter2 = ax2.scatter(pc_df['ALSTM_CXSPEED_U_PC_000'], pc_df['ALSTM_CXSPEED_U_PC_001'], pc_df['ALSTM_CXSPEED_U_PC_002'], c=df_speed_cmd)

        # # set axis labels and title
        # ax2.set_xlabel('PC1')
        # ax2.set_ylabel('PC2')
        # ax2.set_zlabel('PC3')
        # colorbar2 = plt.colorbar(scatter2)
        # colorbar2.set_label('forward velocity (m/s)')

        # # show the plot
        # plt.show()






        # import matplotlib.pyplot as plt
        # # create a 3D scatter plot
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.scatter(pc_df['ALSTM_CXSPEED_U_PC_000'], pc_df['ALSTM_CXSPEED_U_PC_001'])

        # # set axis labels and title
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # plt.title('3D Scatter Plot')

        # # show the plot
        # plt.show()




