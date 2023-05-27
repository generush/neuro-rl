# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

from typing import List

import numpy as np
import pandas as pd

from utils.data_processing import process_data
from analysis.analyze_pca import compute_pca, export_pca, export_scl
from plotting.generation import generate_dropdown, generate_graph
from plotting.plot import plot_scatter3_ti_tf
from embeddings.embeddings import Data, Embeddings, MultiDimensionalScalingEmbedding, PCAEmbedding, MDSEmbedding, ISOMAPEmbedding,LLEEmbedding, LEMEmbedding, TSNEEmbedding, UMAPEmbedding

import sklearn.preprocessing
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.ticker import ScalarFormatter

from scipy.optimize import minimize

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk


import os
from pdfCropMargins import crop

def crop_pdfs_in_folder(folder_path):
    # Create the "cropped" folder if it doesn't exist
    cropped_folder = os.path.join(folder_path, "cropped")
    os.makedirs(cropped_folder, exist_ok=True)

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".pdf"):
            crop([file_path, '-o', folder_path + '/cropped/'])
        else:
            print(f"Skipping file: {file_name} (not a PDF)")
            
def export_pca(pca: sklearn.decomposition.PCA, path: str):
    pk.dump(pca, open(path,"wb"))

def import_pca(path: str):
    return pk.load(open(path,'rb'))

def transform(df_raw, tf):
    X_raw = df_raw.to_numpy()
    scaler = sklearn.preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_transformed = np.matmul(X_scaled, tf)
    return pd.DataFrame(X_transformed)
    
# define the objective function to minimize
def objective(d, x, v):
    return -np.linalg.norm(np.matmul(x, d))

# define the constraint function that enforces ||d|| = 1
def constraint(d):
    return np.linalg.norm(d) - 1

def interpolate_data(data, speed_cmd, tangling, dt):
    
    N_INTERP_TIMES = 1000

    unique_speeds = np.unique(speed_cmd)

    N_CONDITIONS = len(unique_speeds)

    interp_x = np.zeros((N_INTERP_TIMES, N_CONDITIONS))
    interp_y = np.zeros((N_INTERP_TIMES, N_CONDITIONS))
    interp_z1 = np.zeros((N_INTERP_TIMES, N_CONDITIONS))
    interp_z2 = np.zeros((N_INTERP_TIMES, N_CONDITIONS))
    interp_s = np.zeros((N_INTERP_TIMES, N_CONDITIONS))
    interp_t = np.zeros((N_INTERP_TIMES, N_CONDITIONS))

    unique_speeds = np.unique(speed_cmd)
    for idx, v in enumerate(unique_speeds):
        mask = (speed_cmd == v)
        x, y, z1, z2 = data[mask, 0], data[mask, 1], data[mask, 2], data[mask, 3]
        s = speed_cmd[mask]
        t = tangling[mask]

        n = len(x)
        time = np.arange(0, n + 1) * dt
        time_periodic = np.arange(0, len(x) + 1) * dt

        x = np.append(x, x[0])
        y = np.append(y, y[0])
        z1 = np.append(z1, z1[0])
        z2 = np.append(z2, z2[0])
        t = np.append(t, t[0])
        s = np.append(s, s[0])

        spline_x = CubicSpline(time_periodic, x, bc_type='periodic')
        spline_y = CubicSpline(time_periodic, y, bc_type='periodic')
        spline_z1 = CubicSpline(time_periodic, z1, bc_type='periodic')
        spline_z2 = CubicSpline(time_periodic, z2, bc_type='periodic')
        spline_t = CubicSpline(time_periodic, t, bc_type='periodic')
        spline_s = CubicSpline(time_periodic, s, bc_type='periodic')

        fine_time = np.linspace(time[0], time[-1], num=N_INTERP_TIMES)

        interp_x_vals = spline_x(fine_time)
        interp_y_vals = spline_y(fine_time)
        interp_z1_vals = spline_z1(fine_time)
        interp_z2_vals = spline_z2(fine_time)
        interp_t_vals = spline_t(fine_time)
        interp_s_vals = spline_s(fine_time)

        interp_x[:,idx] = interp_x_vals
        interp_y[:,idx] = interp_y_vals
        interp_z1[:,idx] = interp_z1_vals
        interp_z2[:,idx] = interp_z2_vals
        interp_s[:,idx] = interp_s_vals
        interp_t[:,idx] = interp_t_vals

    return interp_x, interp_y, interp_z1, interp_z2, interp_s, interp_t

def plot_data(x_data, y_data, z_data, c_data, cc_global_min, cc_global_max, data_type, zlabel, clabel, cmap, path, save_figs = False):
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    xx = np.concatenate(x_data, axis=0)
    yy = np.concatenate(y_data, axis=0)
    zz = np.concatenate(z_data, axis=0)
    cc = np.concatenate(c_data, axis=0)

    # plot figures with speed colors and tangling colors
    scatter1 = ax.scatter(xx, yy, zz, c=cc, s=2*np.ones_like(xx), cmap=cmap, vmin=cc_global_min, vmax=cc_global_max, alpha=1, depthshade=True, rasterized=True)
    scatter2 = ax.scatter(xx.flatten(), yy.flatten(), 1.5 * zz.min()*np.ones_like(zz), c='grey', s=1*np.ones_like(xx), alpha=1, depthshade=True, rasterized=True)

    # Add labels and a legend
    xlabel = 'PC 1'
    ylabel = 'PC 2'
    ax.set_xlabel('PC 1')
    ax.set_ylabel('PC 2')
    ax.set_zlabel(zlabel)

    manager = plt.get_current_fig_manager()
    manager.window.title(data_type + ' /// ' + xlabel + ylabel + zlabel + ' /// ' + clabel + ' /// ' + ' interpolated average cycles')

    ax.view_init(20, 55)

    norm = mcolors.Normalize(vmin=cc_global_min, vmax=cc_global_max)
    cbar = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, shrink=0.75)
    cbar.set_label(clabel, labelpad=-29, y=1.1, rotation=0)

    cax = cbar.ax
    cax.set_position([0.80, 0.15, 0.02, 0.5])

    if save_figs:
        filename = f"{data_type}__{xlabel}{ylabel}{zlabel}__{clabel}".replace("/", "-")
        fig.savefig(path + filename + '.pdf', format='pdf', dpi=600, facecolor=fig.get_facecolor())

def analyze_pca_speed_axis(path: str, data_names: List[str], file_suffix: str = ''):
    N_COMPONENTS = 12

    # load DataFrame
    df = process_data(path + 'RAW_DATA' + file_suffix + '.csv')
    dt = df['TIME'][1].compute().to_numpy() - df['TIME'][0].compute().to_numpy()

    x_data = []
    y_data = []
    z1_data = []
    z2_data = []
    s_data = []
    t_data = []

    for idx, data_type in enumerate(data_names):
        # get data for PCA analysis (only raw data)
        filt_data = df.loc[:, df.columns.str.contains(data_type + '_RAW')].compute()
        tangl_data = df.loc[:, df.columns.str.contains(data_type + '_TANGLING')].compute()
        spd_cmd = df.loc[:, 'OBS_RAW_009_u_star'].compute()
        spd_act = df.loc[:, 'OBS_RAW_000_u'].compute()
        idx = df.index.compute()

        df_neuron = filt_data.loc[idx].reset_index(drop=True)
        df_speed_cmd = spd_cmd.loc[idx].reset_index(drop=True)
        df_speed_act = spd_act.loc[idx].reset_index(drop=True)
        df_tangling = tangl_data.loc[idx].reset_index(drop=True)

        N_DIMENSIONS = len(df_neuron.columns)
        COLUMNS = np.char.mod(data_type + 'SPEED_PC_%03d', np.arange(N_COMPONENTS))

        if N_DIMENSIONS > 0:
            scl, pca, pc_df = compute_pca(df_neuron, N_COMPONENTS, COLUMNS)
            export_scl(scl, path + data_type + '_SPEED_SCL.pkl')
            export_pca(pca, path + data_type + '_SPEED_PCA.pkl')
            pc_df.to_csv(path + data_type + '_SPEED_PC_DATA' + file_suffix + '.csv')
        
        x_s = pc_df.iloc[:, 2:]
        v_bar = df_speed_cmd.unique()
        NUM_SPEEDS = len(v_bar)

        x_bar = np.zeros((NUM_SPEEDS, N_COMPONENTS - 2), dtype=float)
        for i in range(NUM_SPEEDS):
            x_bar[i, :] = x_s[df_speed_cmd == v_bar[i]].mean().to_numpy()

        d0 = np.ones((x_s.shape[1], 1)) / x_s.shape[1]
        problem = {
            'fun': objective,
            'x0': d0,
            'args': (x_bar, v_bar - v_bar.mean()),
            'constraints': {'type': 'eq', 'fun': constraint}
        }

        result = minimize(**problem)
        d_opt = result.x

        speed_axis = np.matmul(pca.components_[2:].transpose(), d_opt)

        pc_12speed_tf = np.column_stack((
            pca.components_[0],
            pca.components_[1],
            speed_axis
        ))

        pc_123_tf = np.column_stack((
            pca.components_[0],
            pca.components_[1],
            pca.components_[2],
        ))

        pc_12speed_df = transform(df_neuron, pc_12speed_tf)
        pc_123_df = transform(df_neuron, pc_123_tf)

        pc_pc2_pc3_speedaxis_df = pd.concat([pc_123_df, pc_12speed_df.iloc[:, -1]], axis=1)

        # Interpolated data
        x, y, z1, z2, s, t = interpolate_data(pc_pc2_pc3_speedaxis_df.to_numpy(), df_speed_cmd.to_numpy(), df_tangling.to_numpy(), dt)
        x_data.append(x)
        y_data.append(y)
        z1_data.append(z1)
        z2_data.append(z2)
        s_data.append(s)
        t_data.append(t)

        # export some data for the paper/suppl matl
        pd.DataFrame(pc_12speed_tf).to_csv(path + 'info_' + data_type + '_pc_12speed_tf.csv')
        pd.DataFrame(pc_123_tf).to_csv(path + 'info_' + data_type + '_pc_123_tf.csv')
        pd.DataFrame(pca.explained_variance_ratio_.cumsum()).to_csv(path + 'info_' + data_type + '_cumvar.csv')
        pd.DataFrame(x).to_csv(path + 'info_' + data_type + '_x_by_speed.csv')
        pd.DataFrame(y).to_csv(path + 'info_' + data_type + '_y_by_speed.csv')
        pd.DataFrame(z1).to_csv(path + 'info_' + data_type + '_z1_by_speed.csv')
        pd.DataFrame(z2).to_csv(path + 'info_' + data_type + '_z2_by_speed.csv')
        pd.DataFrame(t).to_csv(path + 'info_' + data_type + '_tangling_by_speed.csv')
        pd.DataFrame(d_opt).to_csv(path + 'info_' + data_type + '_dopt.csv')
        export_scl(scl, path + 'info_' + data_type +'_SCL' + '.pkl')
        export_pca(pca, path + 'info_' + data_type +'_PCA' + '.pkl')

    s_global_min = np.min(s_data)
    s_global_max = np.max(s_data)
    t_global_min = np.min(t_data)
    t_global_max = np.max(t_data)

    pd.DataFrame(pca.explained_variance_ratio_.cumsum()).to_csv(data_type + '_cumvar.csv')


    # Plot the data for each data_type
    for idx, data_type in enumerate(data_names):
        xx = x_data[idx]
        yy = y_data[idx]
        zz1 = z1_data[idx]
        zz2 = z2_data[idx]
        ss = s_data[idx]
        tt = t_data[idx]
        
        # PC1, PC2, PC3, Speed
        plot_data(xx, yy, zz1, ss, s_global_min, s_global_max, data_type, 'PC 3', 'u [m/s]', 'Spectral', path, save_figs=True)

        # PC1, PC2, PC3, Tangling
        plot_data(xx, yy, zz1, tt, t_global_min, t_global_max, data_type, 'PC 3', 'Tangling', 'viridis', path, save_figs=True)

        # PC1, PC2, SpeedAxis, Speed
        plot_data(xx, yy, zz2, ss, s_global_min, s_global_max, data_type, 'Speed Axis', 'u [m/s]', 'Spectral', path, save_figs=True)

        # PC1, PC2, SpeedAxis, Speed
        plot_data(xx, yy, zz2, tt, t_global_min, t_global_max, data_type, 'Speed Axis', 'Tangling', 'viridis', path, save_figs=True)

    plt.show()

    # crop white space out of pdfs
    crop_pdfs_in_folder(path)
    
    print('done plotting')