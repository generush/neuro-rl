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
a.rename(columns={'Subfolder_4':'forceY'}, inplace=True)
a.rename(columns={'Subfolder_5':'model_name'}, inplace=True)


# Further filter for rows starting with the base path, if needed
a = a[a['run_type'] == 'find_most_robust_model_diffgaitcriteriaRH']

# Parse 'model_name' to extract 'episode_number' and 'reward'
# Make sure 'model_name' is treated as a string column
# Note: The 'model_name' column might have leading/trailing whitespaces that need to be stripped
a['model_name'] = a['model_name'].str.strip()# Parse 'model_name' to extract 'episode_number' and handle NaN values by using Pandas Nullable Integer Data Type
a['episode_number'] = a['model_name'].str.extract(r'ep_(\d+)_')[0].astype('Int64')
a['reward'] = a['model_name'].str.extract(r'rew_([-\d.]+)\.pth')[0].astype(float)

# Drop rows where 'reward' is NaN
a.dropna(subset=['reward'], inplace=True)

# Convert 'Recoveries' and 'Trials' columns to integers
a['Recoveries'] = a['Recoveries'].astype(int)
a['Trials'] = a['Trials'].astype(int)

# Create the 'robustness' column as the ratio of 'Recoveries' to 'Trials'
a['Robustness'] = a['Recoveries'] / a['Trials']

import matplotlib.pyplot as plt
import pandas as pd

# Assuming 'a' is your DataFrame that includes all data

# Define your array of forceY values
forceY_values = [-2.0, -2.5]

# Get unique model types from the entire DataFrame
model_types = a['model_type'].unique()

### PLOT REWARD

# Create a figure for the Reward plot
fig, ax = plt.subplots(1, 1, figsize=(10, 6))  # 1 row, 1 column

for model_type in model_types:
    for forceY in forceY_values:
        # Filter the DataFrame for the current model type and forceY value
        model_df = a[(a['model_type'] == model_type) & (a['forceY'] == str(forceY))].sort_values('episode_number')
        
        # Plot the line for this model type and forceY value on the subplot
        ax.plot(model_df['episode_number'], model_df['reward'], label=f'{model_type} (forceY={forceY})')

# Setting title, labels, and legend for the Reward plot
ax.set_title('Reward Over Episodes by Model Type')
ax.set_xlabel('Episode Number')
ax.set_ylabel('Reward')
ax.legend(title='Model Type & forceY')

plt.tight_layout()

### PLOT ROBUSTNESS

# Create a figure for the Robustness plot with subplots for each model type
fig, axes = plt.subplots(len(model_types), 1, figsize=(10, 6 * len(model_types)), sharex=True, constrained_layout=True)

for i, model_type in enumerate(model_types):
    for forceY in forceY_values:
        # Filter and sort the DataFrame for the current model type and forceY value
        model_df = a[(a['model_type'] == model_type) & (a['forceY'] == str(forceY))].sort_values('episode_number')
        
        # Plot the line for this model type and forceY value on the corresponding subplot
        axes.plot(model_df['episode_number'], model_df['Robustness'], label=f'{model_type} (forceY={forceY})')

    # Adjust the title, labels, and legend for each subplot
    axes.set_title(f'Model Type: {model_type}', pad=5)
    axes.set_ylabel('Robustness')
    axes.legend()

# Set common labels
plt.xlabel('Episode Number')
plt.show()





print('hi')
