import os
import pandas as pd
import matplotlib.pyplot as plt

a = pd.read_csv('aggregated_data.csv')

# Define the base path that we want to filter by and remove
base_path = '../../neuro-rl/neuro_rl/data/raw/'

# Further filter for rows starting with the base path, if needed
a = a[a['FolderPath'].str.startswith(base_path)]

# Remove the base path from the 'FolderPath' column, if present
a['FolderPath'] = a['FolderPath'].str.replace(base_path, '', regex=False)

# Now split the 'FolderPath' column by '/'
split_columns = a['FolderPath'].str.split('/', expand=True)

# Add the split columns to the DataFrame, you can name these columns as you like
for i, column in enumerate(split_columns.columns):
    a[f'Subfolder_{i+1}'] = split_columns[column]

# Add the split columns to the DataFrame with the specific names given
a.rename(columns={'Subfolder_1':'model_type'}, inplace=True)
a.rename(columns={'Subfolder_2':'run_type'}, inplace=True)
a.rename(columns={'Subfolder_3':'steps_after_stance_begins'}, inplace=True)
a.rename(columns={'Subfolder_4':'length_s'}, inplace=True)
a.rename(columns={'Subfolder_5':'forceY'}, inplace=True)
a.rename(columns={'Subfolder_6':'model_name'}, inplace=True)

# Make sure 'steps_after_stance_begins' is numeric
a['steps_after_stance_begins'] = pd.to_numeric(a['steps_after_stance_begins'], errors='coerce')

# Further filter for rows starting with the base path, if needed
a = a[a['run_type'] == 'final_model_robustness']

a = a[a['model_type'] == 'ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR']

# Parse 'model_name' to extract 'episode_number' and 'reward'
# Make sure 'model_name' is treated as a string column
# Note: The 'model_name' column might have leading/trailing whitespaces that need to be stripped
a['model_name'] = a['model_name'].str.strip()# Parse 'model_name' to extract 'episode_number' and handle NaN values by using Pandas Nullable Integer Data Type
a['episode_number'] = a['model_name'].str.extract(r'ep_(\d+)_')[0].astype('Int64')
a['reward'] = a['model_name'].str.extract(r'rew_([-\d.]+)\.pth')[0].astype(float)

# Drop rows where 'reward' is NaN
a.dropna(subset=['reward'], inplace=True)

# Convert 'forceY' and 'length_s' columns from string to float
a['forceY'] = a['forceY'].astype(float)
a['length_s'] = a['length_s'].astype(float)

# Convert 'Recoveries' and 'Trials' columns to integers
a['Recoveries'] = a['Recoveries'].astype(int)
a['Trials'] = a['Trials'].astype(int)

# Create the 'robustness' column as the ratio of 'Recoveries' to 'Trials'
a['Robustness'] = a['Recoveries'] / a['Trials']


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import numpy as np

# Assuming 'a' DataFrame is already loaded and processed

# Get unique model types from the entire DataFrame
model_types = a['model_type'].unique()
model_names = a['model_name'].unique()

all_steps_after_stance_begins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25])
all_forceY = np.array([-12, -3, -1])
# Convert 'forceY' and 'length_s' columns from string to float, if not already
a['forceY'] = a['forceY'].astype(float)
a['length_s'] = a['length_s'].astype(float)

# Convert 'Recoveries' and 'Trials' columns to integers
a['Recoveries'] = a['Recoveries'].astype(int)
a['Trials'] = a['Trials'].astype(int)

# Create the 'Robustness' column as the ratio of 'Recoveries' to 'Trials'
a['Robustness'] = a['Recoveries'] / a['Trials']


a = a.sort_values(by='steps_after_stance_begins')

# Get unique forceY values and model names
forceY_values = sorted(a['forceY'].unique())

# Get unique model names and sort them alphabetically
model_names = sorted(a['model_name'].unique(), reverse=True)


# Determine the layout of subplots based on the number of forceY values
n_rows = 1
n_cols = 3

plt.figure(figsize=(5 * n_cols, 5 * n_rows))

for i, forceY in enumerate(forceY_values, 1):
    ax = plt.subplot(n_cols, n_rows, i)
    for model_name in model_names:
        # Filter the data for the current model_name and forceY value
        df_filtered = a[(a['model_name'] == model_name) & (a['forceY'] == forceY)]

        # Plotting
        ax.plot(df_filtered['steps_after_stance_begins'], df_filtered['Robustness'], label=model_name)

    ax.set_title(f'ForceY: {forceY}')
    ax.set_xlabel('Steps After Stance Begins')
    ax.set_ylabel('Robustness')
    # Only add legend to the first subplot for clarity, or adjust as needed
    if i == 1:
        ax.legend()

plt.tight_layout()
plt.show()

print('hi')