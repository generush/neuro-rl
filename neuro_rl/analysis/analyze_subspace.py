# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

import numpy as np
import pandas as pd
import pickle as pk
import dask.dataframe as dd

import sklearn.decomposition

# FOLDER_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data_PAPER_test_subspace'
FOLDER_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/'

# load DataFrame
df = dd.read_csv(FOLDER_PATH + 'RAW_DATA_AVG' + '.csv')

def V(R, W):
    return 1 - np.linalg.norm( R - np.matmul ( R, np.matmul( W, W.transpose() ) ) ) / np.linalg.norm(R)

DATASETS = [
    # 'OBS',
    # 'ACT',
    # 'ALSTM_HX',
    'ALSTM_CX',
    # 'CLSTM_HX',
    # 'CLSTM_CX',
    # 'AGRU_HX',
    # 'CGRU_HX',
]

NUM_CONDITIONS = int(df['CONDITION'].max().compute() + 1)
R = []
W = []

# loop through each datatype
for i, data_type in enumerate(DATASETS):

    # get data for ith datatype
    df_filt = df.loc[:,df.columns.str.contains(data_type + '_RAW')].compute()

    # loop through each condition
    for j in range(NUM_CONDITIONS):

        # get indices for jth condition
        idx = df.loc[df['CONDITION'] == j].index.compute()

        # get R (single cycle data) matrix: data from ith datatype, jth condition
        RR = df_filt.loc[idx].to_numpy()

        # initialize PCA object
        PPpca = sklearn.decomposition.PCA(n_components=10)

        # fit pca transform
        data_pc = pca.fit_transform(RR)

        # get W (principal components) matrix: data from ith datatype, jth condition
        WW = pca.components_.transpose()

        R.append(RR)
        W.append(WW)

        print('Finished ', data_type, ' CONDITION: ', j)

# initialize subspace overlap matrix:
subspace_overlap = np.zeros((NUM_CONDITIONS, NUM_CONDITIONS), dtype=float)

# loop through each permutation of 2 conditions and compute subspace overlap!
for r in range(NUM_CONDITIONS): # loop through reference cycles
    for c in range(NUM_CONDITIONS): # loop through comparison cycles
        subspace_overlap[r,c] = V(R[c], W[r]) / V(R[c], W[c])
        print('Subspace Overlap for CONDITIONS: ', r, c, ': ', subspace_overlap[r,c])

pd.DataFrame(subspace_overlap).to_csv('subspace_overlap.csv')

# just grab the subpspace overlaps that compare to the first CONDITION (CONDITION 0 vs CONDITION 0, 1, ..., N)
arr_1d = subspace_overlap[0,:]

# reshape it so its in a grid form
arr_3d = arr_1d.reshape((6,6,6))

import numpy as np
from pylab import *
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(8,6))

ax = fig.add_subplot(111,projection='3d')

data = np.random.random(size=(6, 6, 6))

zs, xs, ys = data.nonzero()
xs = -1 + 2 *xs / 5
ys = -1 + 2 *ys / 5
zs = -1 + 2 *zs / 5

# xs = np.linspace(-1, 1, 6)
# ys = np.linspace(-1, 1, 6)
# zs = np.linspace(-1, 1, 6)

colmap = cm.ScalarMappable(cmap='viridis')
colmap.set_array(arr_3d)
colmap.set_clim(vmin=arr_3d.min(), vmax=arr_3d.max()) # set the colorbar limits

yg = ax.scatter(xs, ys, zs, c=arr_3d, cmap='viridis', alpha=1.0)
cbar = fig.colorbar(colmap)

ax.set_xlabel('u')
ax.set_ylabel('v')
ax.set_zlabel('r')

plt.show()


# Compute the cosine similarity between the first PC for different CONDITIONS (000, 001):
sklearn.metrics.pairwise.cosine_similarity(W[0][:,0].reshape(1,-1) , W[1][:,0].reshape(1,-1) )

print('hi')