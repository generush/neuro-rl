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

    N_COMPONENTS = 5

    # load DataFrame
    df = process_data(path + 'RAW_DATA' + file_suffix + '.csv')

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
        # create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc_speed_df[0], pc_speed_df[1], pc_speed_df[2])

        # set axis labels and title
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title('3D Scatter Plot')

        # show the plot
        plt.show()



        import matplotlib.pyplot as plt
        # create a 3D scatter plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pc_df['ALSTM_CXSPEED_U_PC_000'], pc_df['ALSTM_CXSPEED_U_PC_001'], pc_df['ALSTM_CXSPEED_U_PC_002'])

        # set axis labels and title
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.title('3D Scatter Plot')

        # show the plot
        plt.show()



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




