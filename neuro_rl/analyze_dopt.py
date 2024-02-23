# https://plotly.com/python/3d-scatter-plots/
import logging
from collections import OrderedDict

import dash_bootstrap_components as dbc
from dash import Dash, Input, Output, dcc, html

import numpy as np
import pandas as pd
import pickle as pk

import sklearn.decomposition

import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm

FILE_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/supp/exp3_modspeeds/info_A_LSTM_CX_dopt_compare.csv'

# load DataFrame
arr = pd.read_csv(FILE_PATH, header=None, index_col=None).to_numpy()

# Replace these with your actual dimension and condition labels
dimensions = ['PC 3', 'PC 4', 'PC 5', 'PC 6', 'PC 7', 'PC 8', 'PC 9', 'PC 10', 'PC 11', 'PC 12']
conditions = ['u = 0.8 to 2 m/s', 'u = -0.8 to -2 m/s', 'v = 0.8 to 2 m/s', 'v = -0.8 to -2 m/s', 'u = 2 m/s, r = 0.1 to 0.25 rad/s', 'u = 2 m/s, r = -0.1 to -0.25 rad/s']


bar_width = 0.10  # This will determine the width of the bars. Adjust as needed
spacing = 0.02  # This will determine the space between the bars. Adjust as needed

x = np.arange(len(dimensions))  # the label locations

fig, ax = plt.subplots(figsize=(10, 6))

# Get the colormap and create an array of colors, one for each condition
cmap = cm.get_cmap('viridis')  # Replace 'viridis' with any colormap you like
# colors = [cmap(i) for i in np.linspace(0, 1, arr.shape[1])]
hatches = ['', '///', '', '///', '', '///']  # Hatches for light red, light blue, and light green

colors = ['#E53935', '#FF8A80', '#2979FF', '#80D8FF', '#43A047', '#A5D6A7']
colors = ['#E53935', '#E53935', '#2979FF', '#2979FF', '#43A047', '#43A047']
# Create a bar for each condition
for i in range(arr.shape[1]):
    ax.bar(x + i*(bar_width+spacing), arr[:, i], bar_width, label=conditions[i], color=colors[i], hatch=hatches[i])
    # bar.set_hatch_linewidth(0.7)  # Adjust hatch linewidth to make it less coarse

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Contribution of PCs to Speed Axes', rotation=90)
ax.set_xticks(x + bar_width / 2 + (arr.shape[1]-1)*spacing / 2)  # Adjust x-tick locations
ax.set_xticklabels(dimensions)
ax.legend()

# Draw x-axis line
ax.axhline(0, color='black')

# Rotate x-axis labels for better visibility if they are long
plt.xticks()

fig.tight_layout()
plt.show()

# Save the plot as an SVG file
plt.savefig('d_opt_barchart.svg', format='svg')
plt.savefig('d_opt_barchart.pdf', format='pdf', dpi=600)
plt.savefig('d_opt_barchart.png', format='png', dpi=600)

print('hi')