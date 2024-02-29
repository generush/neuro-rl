# https://plotly.com/python/3d-scatter-plots/

from typing import List

import numpy as np
import pandas as pd

from analysis.analyze_pca import compute_pca
from utils.io import export_pk

import sklearn.preprocessing
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

import matplotlib
matplotlib.use('TkAgg')  # Replace 'TkAgg' with another backend if needed

from scipy.optimize import minimize

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk

from constants.normalization_types import NormalizationType
            
def transform(X_raw, tf):
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

def append_speed_axis(df: pd.DataFrame, data_names: List[str], max_dims: int, norm_type: NormalizationType, export_path: str):

    dt = df['TIME'][1] - df['TIME'][0]

    x_data = []
    y_data = []
    z1_data = []
    z2_data = []
    s_data = []
    t_data = []

    spd_cmd = df.loc[:, 'OBS_RAW_009_u_star']
    spd_act = df.loc[:, 'OBS_RAW_000_u']
    
    for idx, data_type in enumerate(data_names):

        # get data for PCA analysis (only raw data)
        filt_data = df.loc[:, df.columns.str.contains(data_type + '_RAW')]
        tangl_data = df.loc[:, df.columns.str.contains(data_type + '_TANGLING')]
        idx = df.index

        df_neuron = filt_data.loc[idx].values
        df_speed_cmd = spd_cmd.loc[idx].values
        df_speed_act = spd_act.loc[idx].values
        df_tangling = tangl_data.loc[idx].values
        

        
        n_components = min(len(filt_data.columns), max_dims)
        column_names_pc = np.char.mod(data_type + 'SPEED_PC_%03d', np.arange(n_components))

        if n_components > 0:
            scl, pca, data_pc = compute_pca(df_neuron, n_components, norm_type)

            # create DataFrame
            df_pc = pd.DataFrame(data_pc, columns=column_names_pc)

            # export_pk(scl, data_type + '_SCL.pkl')
            # export_pk(pca, data_type + '_PCA.pkl')
            # df_pc.to_csv(path + data_type + '_SPEED_PC_DATA' + file_suffix + '.csv')
        
        x_s = df_pc.iloc[:, 2:]
        v_bar = np.unique(df_speed_cmd)
        n_speeds = len(v_bar)

        x_bar = np.zeros((n_speeds, n_components - 2), dtype=float)
        for i in range(n_speeds):
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


        # export some data for the paper/suppl matl
        pd.DataFrame(pc_12speed_tf).to_csv(export_path + 'info_' + data_type + '_pc_12speed_tf.csv')
        pd.DataFrame(pc_123_tf).to_csv(export_path + 'info_' + data_type + '_pc_123_tf.csv')
        # pd.DataFrame(pca.explained_variance_ratio_.cumsum()).to_csv(path + 'info_' + data_type + '_cumvar.csv')

        speedaxis = pc_12speed_df.iloc[:, 2].values


        # Append tangling as a new column in the data DataFrame
        df_speed_axis = pd.DataFrame(speedaxis, columns = [data_type + '_SPEED_AXIS'])
        df = pd.concat([df, df_speed_axis], axis=1)

    return df