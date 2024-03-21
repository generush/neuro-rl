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


# Further filter for rows starting with the base path, if needed
a = a[a['run_type'] == 'final_model_robustness']

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

all_steps_after_stance_begins = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])
all_forceY = np.array([-12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12])

for i, model_type in enumerate(model_types):
    # Create a new figure for each model type
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(f'Model Type: {model_type}', pad=20, fontsize=14)
    
    model_df = a[a['model_type'] == model_type]

    # Ensure there's data for this model_type before proceeding
    if not model_df.empty:
        pivot_table = model_df.pivot_table(index='steps_after_stance_begins', columns='forceY', values='Robustness', aggfunc='mean')
        Z = pivot_table.values

        # Handling cases where pivot result might have NaNs due to missing combinations in data
        Z[np.isnan(Z)] = 0
        
        X, Y = np.meshgrid(all_forceY, all_steps_after_stance_begins)

        # Smooth the data
        f = interpolate.interp2d(all_forceY, all_steps_after_stance_begins, Z, kind='linear')
        x_smooth = np.linspace(all_forceY.min(), all_forceY.max(), 100)
        y_smooth = np.linspace(all_steps_after_stance_begins.min(), all_steps_after_stance_begins.max(), 100)
        X_smooth, Y_smooth = np.meshgrid(x_smooth, y_smooth)
        Z_smooth = f(x_smooth, y_smooth)

        # Create the surface plot
        orig_map = plt.cm.get_cmap('plasma')
        reversed_map = orig_map.reversed()
        surf = ax.plot_surface(X_smooth, Y_smooth, Z_smooth, cmap=reversed_map, vmin=0, vmax=1, linewidth=0, edgecolor='none', alpha=1.0)

        # Set labels for axes
        ax.set_xlabel('ForceY')
        ax.set_ylabel('Steps After Stance Begins')
        ax.set_zlabel('Robustness')
        
    else:
        print(f"No data available for model type: {model_type}")

# Show the plot for this model type
plt.show()

print('hi')











import torch
# model = torch.load('/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/models/A1-1.0MASS-LSTM16-TERR-01/nn/last_A1Terrain_ep_4600_rew_16.256865.pth')

# model = torch.load('/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/models/ANYMAL-1.0MASS-LSTM16-TERR-01/nn/last_AnymalTerrain_ep_2000_rew_18.73817.pth')

model = torch.load('/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/models/ANYMAL-1.0MASS-LSTM4-CORLDISTTERR/nn/last_AnymalTerrain_ep_6700_rew_20.21499.pth')


state_dict = {key.replace('a2c_network.a_rnn.rnn.', ''): value for key, value in model['model'].items() if key.startswith('a2c_network.a_rnn.rnn')}



a = state_dict['bias_ih_l0']
b = state_dict['bias_hh_l0']
c = model['model']['a2c_network.actor_mlp.0.weight'][:,176:]











# Load your DataFrame 'a' here
# a = pd.read_csv('your_data.csv')

# Assuming 'a' is already prepared as per your initial code

# Define your array of forceY values for low, mid, and high categories



low_impulse_forceY = [-0.333, -1, -4]
mid_impulse_forceY = [-0.667, -2, -8]
high_impulse_forceY = [-1, -3, -12]

low_impulse_length_s = [0.4, 0.1, 0.02]
mid_impulse_length_s = [0.4, 0.1, 0.02]
high_impulse_length_s = [0.4, 0.1, 0.02]

# Combine all forceY values and length_s values for ease of use
all_forceY = [low_impulse_forceY, mid_impulse_forceY, high_impulse_forceY]
all_length_s = [low_impulse_length_s, mid_impulse_length_s, high_impulse_length_s]
forceY_categories = ['Low', 'Mid', 'High']  # For labeling purposes






# Get unique model types from the entire DataFrame
model_types = a['model_type'].unique()

# Colors for the lines
colors = ['black', 'red', 'blue']

# Create a figure for the Robustness plot with subplots for each model type and forceY category, plus an additional column for the summary
fig, axes = plt.subplots(len(model_types), 4, figsize=(20, 6 * len(model_types)), sharex=True, constrained_layout=True)

# List of markers to cycle through, including an 'o' for unfilled circle
markers = [('o', 'none'), ('x', None), ('+', None)]

# Disturbance duration labels for the legend
disturbance_labels = ['20ms disturbance', '100ms disturbance', '400ms disturbance']

for i, model_type in enumerate(model_types):
    # Set the model_type title in the first column for each row
    axes[i, 0].set_title(f'Model Type: {model_type.replace("-CONDENSED", "")}', pad=5, fontsize=10)
    
    for k, (forceY_values, length_s_values) in enumerate(zip(all_forceY, all_length_s)):
        for j, (forceY, length_s) in enumerate(zip(forceY_values, length_s_values)):
            marker, facecolor = markers[j % len(markers)]
            markerfacecolor = 'none' if facecolor == 'none' else 'auto'
            line_color = colors[j % len(colors)]

            model_df = a[(a['model_type'] == model_type) & (a['forceY'] == forceY) & (a['length_s'] == length_s)].sort_values('episode_number')

            ax = axes[i, k]
            
            ax.plot(model_df['episode_number'], model_df['Robustness'], marker=marker, linestyle='--', color=line_color, markerfacecolor=markerfacecolor, label=disturbance_labels[j])
            
            # Set y-axis limits to 0 to 1 for all subplots
            ax.set_ylim(0, 1)

    # Summary plot in the 4th column for each model type
    summary_ax = axes[i, 3]
    summary_ax.set_title('Summary', pad=5, fontsize=10)

    # Example: Aggregating with mean for simplicity. Replace with your weighted aggregation method
    summary_df = a[a['model_type'] == model_type].groupby('episode_number').mean().reset_index()
    
    # Plot the summary line
    summary_ax.plot(summary_df['episode_number'], summary_df['Robustness'], color='black', linestyle='--', label='Summary')
    
    # Add black square markers on all summary points
    summary_ax.plot(summary_df['episode_number'], summary_df['Robustness'], 's', color='black', markerfacecolor='none')

    # Find the episode number with maximum summary robustness
    max_robustness_idx = summary_df['Robustness'].idxmax()
    max_robustness_episode = summary_df.loc[max_robustness_idx, 'episode_number']
    max_robustness_value = summary_df.loc[max_robustness_idx, 'Robustness']

    # Highlight the maximum robustness point with a filled black square
    summary_ax.plot(max_robustness_episode, max_robustness_value, 's', color='black', markerfacecolor='black')

    summary_ax.set_ylim(0, 1)
    summary_ax.legend()

# Set common x-axis label for the entire figure at the bottom, adjusted position
fig.text(0.5, 0.02, 'Episode', ha='center', fontsize=10)

# Set common y-axis label for the entire figure at the side, adjusted position
fig.text(0.005, 0.5, 'Robustness', va='center', rotation='vertical', fontsize=10)

# Adjust layout to make space for the common y-axis label
plt.subplots_adjust(left=0.07, right=0.95, top=0.95, bottom=0.12)

# Create a single legend for the entire figure, placed at the top
fig.legend(labels=disturbance_labels + ['Summary'], loc='upper center', ncol=4, fontsize=9)

plt.show()

for i, model_type in enumerate(model_types):
    # Filter the DataFrame for the current model_type
    model_df = a[a['model_type'] == model_type]

    # Aggregate the data (assuming mean for simplicity, replace with your method)
    summary_df = model_df.groupby('episode_number').mean().reset_index()

    # Find the episode number with the maximum summary robustness
    max_robustness_idx = summary_df['Robustness'].idxmax()
    max_robustness_episode = summary_df.loc[max_robustness_idx, 'episode_number']

    # Find the rows in the original DataFrame that match the model_type and the episode_number of maximum robustness
    matching_rows = model_df[model_df['episode_number'] == max_robustness_episode]

    # Assuming 'model_name' is available in 'matching_rows', and you want to print all unique model_names for the given condition
    unique_model_names = matching_rows['model_name'].unique()

    print(f"Model Type: {model_type}")
    print("Model Names with Max Robustness:")
    for name in unique_model_names:
        print(name)
    print("----------")
    
print('hi')
