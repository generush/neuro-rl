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

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm

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

def interpolate_data(data, speed_cmd, tangling, dt):
    N_INTERP_TIMES = 1000
    NUM_SPEEDS = len(speed_cmd)
    interpolated_data = []

    for idx, v in enumerate(np.unique(speed_cmd)):
        x = data[:, 0][speed_cmd == v]
        y = data[:, 1][speed_cmd == v]
        z = data[:, 2][speed_cmd == v]

        s = speed_cmd[speed_cmd == v]
        s = tangling[speed_cmd == v]

        n = len(x)
        time = np.arange(0, n + 1) * dt
        time_periodic = np.arange(0, len(x) + 1) * dt
        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z = np.append(z, z[0])
        s = np.append(s, s[0])

        interp_x = CubicSpline(time_periodic, x, bc_type='periodic')
        interp_y = CubicSpline(time_periodic, y, bc_type='periodic')
        interp_z = CubicSpline(time_periodic, z, bc_type='periodic')
        interp_s = CubicSpline(time_periodic, s, bc_type='periodic')

        fine_time = np.linspace(time[0], time[-1], num=N_INTERP_TIMES)

        interp_x_vals = interp_x(fine_time)
        interp_y_vals = interp_y(fine_time)
        interp_z_vals = interp_z(fine_time)
        interp_s_vals = interp_s(fine_time)

        interpolated_data.append((interp_x_vals, interp_y_vals, interp_z_vals, interp_s_vals))

    return interpolated_data, interp_s_vals.min(), interp_s_vals.max()

def plot_data(interpolated_data, ss_min, ss_max, speed_cmd, tangling, dt, data_type):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for data in interpolated_data:
        xx, yy, zz, ss = data
        scatter1 = ax.scatter(xx, yy, zz, c=ss, s=2*np.ones_like(xx), cmap='Spectral', vmin=ss_min, vmax=ss_max, alpha=1, rasterized=True)
        scatter2 = ax.scatter(xx.flatten(), yy.flatten(), 1.5 * zz.min()*np.ones_like(zz), c='grey', s=1*np.ones_like(xx), alpha=1, rasterized=True)

    # Add labels and a legend
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')

    manager = plt.get_current_fig_manager()
    manager.window.title(data_type + ' interpolation')

    ax.view_init(20, 55)

    norm = mcolors.Normalize(vmin=ss_min, vmax=ss_max)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap='Spectral'), ax=ax, shrink=0.75)
    cbar.set_label('Speed', labelpad=-29, y=1.1, rotation=0)
    cax = cbar.ax
    cax.set_position([0.80, 0.15, 0.02, 0.5])
    
def analyze_pca_speed_axis(path: str, data_names: List[str], file_suffix: str = ''):

    N_COMPONENTS = 12

    # load DataFrame
    df = process_data(path + 'NORM_DATA' + file_suffix + '.csv')

    dt = df['TIME'][1].compute().to_numpy() - df['TIME'][0].compute().to_numpy()

    interpolated_data_list = []
    ss_min_list = []
    ss_max_list = []

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

        # Interpolate the data
        interpolated_data, ss_min, ss_max = interpolate_data(pc_df.to_numpy(), df_speed_cmd.to_numpy(), df_tangling.to_numpy(), dt)

        # Store the interpolated data and colorbar limits
        interpolated_data_list.append(interpolated_data)
        ss_min_list.append(ss_min)
        ss_max_list.append(ss_max)

    # find min and max values across all data_types (so color bar can be consistent)
    ss_min = np.min(ss_min_list)
    ss_max = np.max(ss_max_list)

    # Plot the data for each data_type
    for idx, data_type in enumerate(data_names):
        interpolated_data = interpolated_data_list[idx]
        
        plot_data(interpolated_data, ss_min, ss_max, df_speed_cmd.to_numpy(), df_tangling.to_numpy(), dt, data_type)

    plt.show()

    print('done plotting')
