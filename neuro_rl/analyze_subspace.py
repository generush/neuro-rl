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

from analysis.analyze_pca import compute_pca

# FOLDER_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data_PAPER_test_subspace'
FOLDER_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/'
FOLDER_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-05-24_10-43-18_u[-1.0,1.0,2]_v[-1.0,1.0,2]_r[-1.0,1.0,2]_n[100]/'


# load DataFrame
df = dd.read_csv(FOLDER_PATH + 'RAW_DATA_AVG' + '.csv')

def V(R, W):
    return 1 - np.linalg.norm( R - np.matmul ( R, np.matmul( W, W.transpose() ) ) ) / np.linalg.norm(R)

def transform(X_raw, tf):
    scaler = sklearn.preprocessing.StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    X_transformed = np.matmul(X_scaled, tf)
    return pd.DataFrame(X_transformed)
    
DATASETS = [
    # 'OBS',
    # 'ACT',
    # 'ALSTM_HX',
    'A_LSTM_CX',
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
        pca = sklearn.decomposition.PCA(n_components=10)

        pca, data_pc = compute_pca(RR, 10, df_filt.columns[:10])

        RR = transform(RR, np.eye(np.shape(RR)[1])).to_numpy()

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
arr_3d = arr_1d.reshape((2,6,6))

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



import numpy as np
import matplotlib.pyplot as plt


# Create a figure and set the figure size
fig = plt.figure(figsize=(5, 5))

# Add white space above the plot
fig.subplots_adjust(top=0.9)

# Add white space to the left of the plot
fig.subplots_adjust(left=0.2)

# Manually label x and y ticks for each cell
plt.xticks(np.arange(8) + 0.5, [
    'u-  v-  r-',
    'u-  v-  r+',
    'u-  v+  r-',
    'u-  v+  r+',
    'u+  v-  r-',
    'u+  v-  r+',
    'u+  v+  r-',
    'u+  v+  r+'], rotation=45
)

# Manually label x and y ticks for each cell
plt.yticks(np.arange(8), [
    'u-  v-  r-',
    'u-  v-  r+',
    'u-  v+  r-',
    'u-  v+  r+',
    'u+  v-  r-',
    'u+  v-  r+',
    'u+  v+  r-',
    'u+  v+  r+']
)

# Move x-axis tick labels to the top
plt.gca().xaxis.tick_top()
plt.gca().xaxis.set_label_position('top')

# Move y-axis tick labels to the top
plt.gca().yaxis.tick_left()
plt.gca().yaxis.set_label_position('left')

# Remove tick marks
plt.tick_params(axis='both', which='both', top=False, right=False, bottom=False, left=False)

# Add text annotations for cell values
for i in range(8):
    for j in range(8):
        value = subspace_overlap[i, j]
        text = '{:.2f}'.format(value)
        color = 'black' if value == 1.0 else 'white'
        plt.text(j, i, text, ha='center', va='center', color=color, fontsize=7)

# Plot the array as a grayscale grid
plt.imshow(subspace_overlap, cmap='gray', vmin=0, vmax=1)
# plt.colorbar()

# Shrink the height of the colorbar
cbar = plt.colorbar(shrink=0.75) # Adjust the y-axis limits to shrink the height
cbar.ax.yaxis.set_ticks_position('left')
cbar.ax.set_ylabel('subspace overlap', rotation=90)
cbar.ax.set_position([0.88, 0.15, 0.04, 0.7]) 

# Display the plot
# plt.show()


# Save the plot as an SVG file
plt.savefig('grayscale_grid.svg', format='svg')
plt.savefig('grayscale_grid.pdf', format='pdf', dpi=600)
plt.savefig('grayscale_grid.png', format='png', dpi=600)


print('hi')