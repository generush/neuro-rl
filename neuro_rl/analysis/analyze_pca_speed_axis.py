# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from typing import List

import numpy as np
import pandas as pd

from utils.data_processing import process_data
from analysis.analyze_pca import compute_pca
from plotting.generation import generate_dropdown, generate_graph
from plotting.plot import plot_scatter3_ti_tf
from embeddings.embeddings import Data, Embeddings, MultiDimensionalScalingEmbedding, PCAEmbedding, MDSEmbedding, ISOMAPEmbedding,LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding

import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics


# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk

def export_pca(pca: sklearn.decomposition.PCA, path: str):
    pk.dump(pca, open(path,"wb"))

def import_pca(path: str):
    return pk.load(open(path,'rb'))

# define the objective function to minimize
def objective(d, x, v):
    return -np.linalg.norm(np.matmul(x, d))

# define the constraint function that enforces ||d|| = 1
def constraint(d):
    return np.linalg.norm(d) - 1

def analyze_pca_speed_axis(path: str, data_names: List[str], file_suffix: str = ''):

    N_COMPONENTS = 12

    # load DataFrame
    df = process_data(path + 'NORM_DATA' + file_suffix + '.csv')

    dt = df['TIME'][1].compute().to_numpy() - df['TIME'][0].compute().to_numpy()

    for idx, data_type in enumerate(data_names):

        # get data for PCA analysis (only raw data)
        filt_data = df.loc[:,df.columns.str.contains(data_type + '_RAW')].compute()

        # get tangling data
        tangl_data = df.loc[:,df.columns.str.contains(data_type + '_TANGLING')].compute()


        spd_cmd = df.loc[:,'OBS_RAW_009_u_star'].compute()
        spd_act = df.loc[:,'OBS_RAW_000_u'].compute()

        # get indices for jth condition
        idx = df.index.compute()

        df_neuron = filt_data.loc[idx].reset_index(drop=True)
        df_speed_cmd = spd_cmd.loc[idx].reset_index(drop=True)
        df_speed_act = spd_act.loc[idx].reset_index(drop=True)
        df_tangling = tangl_data.loc[idx].reset_index(drop=True)

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
            'args': (x_bar, v_bar - v_bar.mean()), # mean-centered mean velocities
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

        # speed command data
        c = df_speed_cmd.to_numpy()

        # Plot the original trajectory and the interpolated trajectory
        fig = plt.figure()
        fig.patch.set_facecolor('white')
        plt.style.use('fivethirtyeight') # fivethirtyeight is name of style
        plt.rcParams['text.usetex'] = True
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['figure.edgecolor'] = 'white'
        
        ax = fig.add_subplot(111, projection='3d')

        color_vals = []  # Store the original interp_s_vals for color bar

        # get data for specified speed cmd
        for idx, v in enumerate(v_bar):
            x = pc_df.iloc[:, 0][c == v].to_numpy()
            y = pc_df.iloc[:, 1][c == v].to_numpy()
            z = pc_df.iloc[:, 2][c == v].to_numpy()

            # color code (speed or tangling)
            s = df_speed_cmd[c == v].to_numpy()
            s = df_tangling[c == v].to_numpy()

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

        # Set background color to white
        ax.set_facecolor('white')

        # Remove grey box around 3D plot
        # ax.xaxis.pane.fill = True
        # ax.yaxis.pane.fill = True
        # ax.zaxis.pane.fill = True

        # # Remove grid lines
        # ax.xaxis.pane.set_edgecolor('white')
        # ax.yaxis.pane.set_edgecolor('white')
        # ax.zaxis.pane.set_edgecolor('white')

        # Plot the interpolated trajectory as scatter with color corresponding to interp_s
        scatter1 = ax.scatter(xx, yy, zz, c=ss, s=2*np.ones_like(xx), cmap='Spectral', alpha=1, rasterized=True)

        # Add labels and a legend
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        # ax.set_title(data_type + " interpolation")

        # Set the color of the 3D panes
        ax.xaxis.pane.fill = ax.yaxis.pane.fill = ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')
    
        manager = plt.get_current_fig_manager()
        manager.window.title(data_type + ' interpolation')

        # Set the elevation (up/down) and the azimuth (left/right)
        ax.view_init(20, 55)

        # Add a 2D projection (x, y)
        scatter2 = ax.scatter(xx.flatten(), yy.flatten(), 1.5 * zz.min()*np.ones_like(zz), c='grey', s=1*np.ones_like(xx), alpha=1, rasterized=True)

        # # Plot the original data points as scatter
        # scatter3 = ax.scatter(x, y, z, c=s, s=50*np.ones_like(x), alpha=1)

        # Define normalization for colorbars
        norm = mcolors.Normalize(vmin=ss.min(), vmax=ss.max())  # Adjust vmin and vmax according to your data

        # Create colorbars for each scatter plot
        cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='Spectral'), ax=ax, shrink=0.75)

        # labelpad is the padding between colorbar and label, adjust it as needed.
        # rotation=0 makes the label horizontal
        cbar.set_label('Speed', labelpad=-29, y=1.1, rotation=0)

        # Adjust colorbar position
        cax = cbar.ax
        cax.set_position([0.80, 0.15, 0.02, 0.5])  # [left, bottom, width, height]

        minimal_formatting = False
        if minimal_formatting:
            # Create some data

                # Turn off grid lines
                ax.grid(False)

                # Set background color to white
                ax.set_facecolor('white')

                # Remove axis tick marks
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])

                # Remove axis labels
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')

                # set pane color to be transparent:
                ax.xaxis.pane.fill = False
                ax.yaxis.pane.fill = False
                ax.zaxis.pane.fill = False

                # Add mini axis symbol
                ax2 = fig.add_axes([0.0, 0.0, 0.2, 0.2], projection='3d')
                ax2.quiver([0], [0], [0], [1], [0], [0], color='k')  # x direction
                ax2.quiver([0], [0], [0], [0], [1], [0], color='k')  # y direction
                ax2.quiver([0], [0], [0], [0], [0], [1], color='k')  # z direction
                ax2.text(1.1, 0, 0, "PC1", color='k')
                ax2.text(0, 1.1, 0, "PC2", color='k')
                ax2.text(0, 0, 1.1, "Speed Axis", color='k')

                # Make the panes and the grid lines transparent
                ax2.xaxis.pane.fill = ax2.yaxis.pane.fill = ax2.zaxis.pane.fill = False
                ax2.grid(False)

                # Make the axes (including the arrows) invisible
                ax2.set_axis_off()

                # Set the limits and the aspect ratio of the plot
                ax2.set_xlim([-1, 1])
                ax2.set_ylim([-1, 1])
                ax2.set_zlim([-1, 1])
                ax2.set_box_aspect([1,1,1])

                # Match the view angles of the mini axis symbol to the main plot
                elev, azim = np.degrees(ax.elev), np.degrees(ax.azim)  # get the current view angles
                ax2.view_init(elev, azim)  # set the view angles of the mini axis symbol

        

        print(data_type, ' Speed Axis PC3-12:', d_opt)

        plt.tight_layout()
        plt.show()
        fig.savefig(data_type + '.svg', format='svg', dpi=600, facecolor=fig.get_facecolor())
        fig.savefig(data_type + '.pdf', format='pdf', dpi=600)




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




