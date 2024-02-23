# %%


from utils.data_processing import filter_by_column_keywords
from analysis.compute_avg_gait_cycle import compute_avg_gait_cycle
from analysis.analyze_pca import compute_pca

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import yaml

# %matplotlib widget

# Load YAML config file
with open('cfg/analyze/analysis.yaml', 'r') as config_file:
    config = yaml.safe_load(config_file)

# Accessing config variables
input_data_path = config['data_paths']['input']
output_data_path = config['data_paths']['output']
dataset_names = config['dataset_names']
norm_type = config['normalization_type']
max_components = config['max_num_principal_components']
tangling_type = config['tangling_type']

# %% Load DataFrame
raw_df = pd.read_parquet(input_data_path + 'RAW_DATA' + '.parquet')
                         
# %% Load DataFrame
filt_df = filter_by_column_keywords(raw_df, dataset_names, 'RAW')

# %% Compute cycle-average and variance datasets from raw dataset
avg_cycle_df, var_cycle_df = compute_avg_gait_cycle(filt_df)

R = []
W = []

n_conditions = avg_cycle_df['CONDITION'].max() + 1

for data_type in dataset_names:

    data_filt = avg_cycle_df.loc[:,avg_cycle_df.columns.str.contains(data_type + '_RAW')].values
    n_dims = min(data_filt.shape[1], max_components)
    pc_columns = [f'{data_type}_PC_{i+1:03d}' for i in range(n_dims)]

    # loop through each condition
    for j in range(n_conditions):
    
        # get indices for jth condition
        idx = avg_cycle_df.loc[avg_cycle_df['CONDITION'] == j].index

        # get R (single cycle data) matrix: data from ith datatype, jth condition
        RR = data_filt[idx,:]

        # initialize PCA object
        scl, pca, data_pc = compute_pca(RR, max_components, norm_type)

        RR = scl.fit_transform(RR)

        # get W (principal components) matrix: data from ith datatype, jth condition
        WW = pca.components_.transpose()

        R.append(RR)
        W.append(WW)

        print('Finished ', data_type, ' CONDITION: ', j)

# initialize subspace overlap matrix:
subspace_overlap = np.zeros((n_conditions, n_conditions), dtype=float)

def V(R, W):
    return 1 - np.linalg.norm( R - np.matmul ( R, np.matmul( W, W.transpose() ) ) ) / np.linalg.norm(R)

# loop through each permutation of 2 condition_labels and compute subspace overlap!
for r in range(n_conditions): # loop through reference cycles
    for c in range(n_conditions): # loop through comparison cycles
        subspace_overlap[r,c] = V(R[c], W[r]) / V(R[c], W[c])
        print('Subspace Overlap for CONDITIONS: ', r, c, ': ', subspace_overlap[r,c])

pd.DataFrame(subspace_overlap).to_csv('subspace_overlap.csv')

# Simplified labels
if n_conditions == 8:
    labels = ['–', '+']
elif n_conditions == 27:
    labels = ['–', '0', '+']
condition_labels = [f'{u}{v}{r}' for u in labels for v in labels for r in labels]

if n_conditions == 8:
    fig, ax = plt.subplots(figsize=(4, 4))
elif n_conditions == 27:
    fig, ax = plt.subplots(figsize=(8, 8))

# Add white space above the plot
fig.subplots_adjust(left=0.2, top=0.9, right=0.8)

# Create a figure and set the figure size
# Plot the array as a grayscale grid
cax = ax.imshow(subspace_overlap, cmap='gray', vmin=0, vmax=1)

# Add colorbar
cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Subspace Overlap')

# Set tick positions
ax.set_xticks(np.arange(len(condition_labels)))
ax.set_yticks(np.arange(len(condition_labels)))

# Set tick labels
ax.set_xticklabels(condition_labels, rotation=45, ha='left')
ax.set_yticklabels(condition_labels)

# Draw gridlines to separate groups
if n_conditions == 8:
    for i in range(1, 4):
        ax.axhline(2*i-0.5, color='lightgrey', lw=0.5)
        ax.axvline(2*i-0.5, color='lightgrey', lw=0.5)
    for i in range(1, 2):
        ax.axhline(4*i-0.5, color='lightgrey', lw=2)
        ax.axvline(4*i-0.5, color='lightgrey', lw=2)
elif n_conditions == 27:
    for i in range(1, 9):
        ax.axhline(3*i-0.5, color='lightgrey', lw=0.5)
        ax.axvline(3*i-0.5, color='lightgrey', lw=0.5)
    for i in range(1, 3):
        ax.axhline(9*i-0.5, color='lightgrey', lw=2)
    ax.axvline(9*i-0.5, color='lightgrey', lw=2)

if n_conditions == 8:
    # Add text annotations for cell values
    for i in range(8):
        for j in range(8):
            value = subspace_overlap[i, j]
            text = '{:.2f}'.format(value)
            color = 'black' if value == 1.0 else 'white'
            plt.text(j, i, text, ha='center', va='center', color=color, fontsize=7)

# Adjust tick parameters
ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

# Set labels
ax.set_xlabel('Reference Condition (u, v, r)')
ax.set_ylabel('Comparison Condition (u, v, r)')
ax.xaxis.set_label_position('top') 

plt.show()

print('hi')