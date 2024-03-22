import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

path1 = '/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/data/raw/ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR/robustness_gradients_analysis/0/0.02/-9/last_AnymalTerrain_ep_3800_rew_20.310041.pth/'


path2 = '/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/data/raw/ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR/robustness_gradients_analysis/0/0.02/-9/last_AnymalTerrain_ep_4100_rew_20.68903.pth/'


# # Load DataFrame
# obs1 = pd.read_csv(path1 + 'obs.csv', header=None)
# obs2 = pd.read_csv(path2 + 'obs.csv', header=None)

# # Load DataFrame
# cn_in1 = pd.read_csv(path1 + 'cn_in.csv', header=None)
# cn_in2 = pd.read_csv(path2 + 'cn_in.csv', header=None)

# # Load DataFrame
# hn_in1 = pd.read_csv(path1 + 'hn_in.csv', header=None)
# hn_in2 = pd.read_csv(path2 + 'hn_in.csv', header=None)

# # Load DataFrame
# hn_out1 = pd.read_csv(path1 + 'hn_out.csv', header=None)
# hn_out2 = pd.read_csv(path2 + 'hn_out.csv', header=None)

# # Load DataFrame
# cn_in_grad1 = pd.read_csv(path1 + 'cn_in_grad.csv', header=None)
# cn_in_grad2 = pd.read_csv(path2 + 'cn_in_grad.csv', header=None)

# # Load DataFrame
# hn_in_grad1 = pd.read_csv(path1 + 'hn_in_grad.csv', header=None)
# hn_in_grad2 = pd.read_csv(path2 + 'hn_in_grad.csv', header=None)

# # Load DataFrame
# hn_out_grad1 = pd.read_csv(path1 + 'hn_out_grad.csv', header=None)
# hn_out_grad2 = pd.read_csv(path2 + 'hn_out_grad.csv', header=None)


raw_df1 = pd.read_parquet(path1 + 'RAW_DATA' + '.parquet')
raw_df2 = pd.read_parquet(path2 + 'RAW_DATA' + '.parquet')


# # find time of disturbance (also at stance), find the time the disturbance response ends (at next stance)
# # compare data in that disturbance response to an equal amount of time steps before the response.

# # or compare two gait cycles before the response to the same time after the response.



# # Assuming 'path1' and 'path2' are defined variables that contain the paths to your data
# # Load your data frames here as per your initial message
# # For demonstration, I'll use dummy data similar to your data frames' structures

# # Creating sample data frames that resemble your actual data frames in structure
# data_frames = {
#     'obs1': obs1,
#     'obs2': obs2,
#     'cn_in1': cn_in1,
#     'cn_in2': cn_in2,
#     'hn_in1': hn_in1,
#     'hn_in2': hn_in2,
#     'hn_out1': hn_out1,
#     'hn_out2': hn_out2,
#     'cn_in_grad1': cn_in_grad1,
#     'cn_in_grad2': cn_in_grad2,
#     'hn_in_grad1': hn_in_grad1,
#     'hn_in_grad2': hn_in_grad2,
#     'hn_out_grad1': hn_out_grad1,
#     'hn_out_grad2': hn_out_grad2,
#     'cn_in_grad_times_values1': cn_in1 * cn_in_grad1,
#     'cn_in_grad_times_values2': cn_in2 * cn_in_grad2,
#     'hn_in_grad_times_values1': hn_in1 * hn_in_grad1,
#     'hn_in_grad_times_values2': hn_in2 * hn_in_grad2,
#     'hn_out_grad_times_values1': hn_out1 * hn_out_grad1,
#     'hn_out_grad_times_values2': hn_out2 * hn_out_grad2
# }

# # Plotting heatmaps for each data frame
# for name, df in data_frames.items():

#     # Transposing the data frame to invert x and y axes
#     df_transposed = df.T
    
#     # Finding the min and max values to set the scale for the color bar
#     vmax = np.max(np.abs(df_transposed.values))
#     vmin = -vmax

#     plt.figure(figsize=(20, 10))
#     plt.imshow(df_transposed, cmap='seismic', vmin=vmin, vmax=vmax)
#     plt.title(f'Heatmap for {name}')
#     plt.xlabel('Columns')
#     plt.ylabel('Rows')

# plt.show()


# print('hi')













import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_data(paths):
    for path in paths:
        # Load DataFrames
        obs = pd.read_csv(path + 'obs.csv', header=None)
        cn_in = pd.read_csv(path + 'cn_in.csv', header=None)
        hn_in = pd.read_csv(path + 'hn_in.csv', header=None)
        hn_out = pd.read_csv(path + 'hn_out.csv', header=None)
        cn_in_grad = pd.read_csv(path + 'cn_in_grad.csv', header=None)
        hn_in_grad = pd.read_csv(path + 'hn_in_grad.csv', header=None)
        hn_out_grad = pd.read_csv(path + 'hn_out_grad.csv', header=None)

        # Calculate derived values
        cn_in_grad_times_values = cn_in * cn_in_grad
        hn_in_grad_times_values = hn_in * hn_in_grad
        hn_out_grad_times_values = hn_out * hn_out_grad

        # List of all DataFrames for plotting
        data_frames = {
            'obs': obs,
            'cn_in': cn_in,
            'hn_in': hn_in,
            'hn_out': hn_out,
            'cn_in_grad': cn_in_grad,
            'hn_in_grad': hn_in_grad,
            'hn_out_grad': hn_out_grad,
            'cn_in_grad_times_values': cn_in_grad_times_values,
            'hn_in_grad_times_values': hn_in_grad_times_values,
            'hn_out_grad_times_values': hn_out_grad_times_values
        }

    return data_frames

def get_response_time_windows(raw_df):
    # Find the first index where 'PERTURB_BEGIN' is 1
    disturb_response_ti = raw_df.loc[raw_df['PERTURB_BEGIN'] == 1].index.values[0]

    # The last nominal response time frame is one less than 'disturb_response_ti'
    nominal_response_tf = disturb_response_ti - 1

    # Initialize 'nominal_response_ti' as an empty dictionary
    nominal_response_ti = {}

    # Find all indices where 'STANCE_BEGIN' is 1, up to 'nominal_response_tf'
    prior_stance_idx = raw_df.loc[:nominal_response_tf][raw_df['STANCE_BEGIN'] == 1].index

    # Check if there are at least two prior entries with 'STANCE_BEGIN' == 1
    if len(prior_stance_idx) >= 2:
        # Take the second-to-last one
        nominal_response_ti = prior_stance_idx[-2]

    # Calculate 'disturb_response_tf'
    disturb_response_tf = disturb_response_ti + nominal_response_tf - nominal_response_ti

    # Return the calculated values
    return disturb_response_ti, nominal_response_tf, nominal_response_ti, disturb_response_tf
    
def plot_data(data_frames, dist_response_ti, nom_response_tf, nom_response_ti, dist_response_tf):
    for name, df in data_frames.items():
        # Focus on the data between dist_response_ti and dist_response_tf
        df_focus = df.iloc[nom_response_ti:dist_response_tf, :]

        # Transposing the focused DataFrame to invert x and y axes
        df_transposed = df_focus.T

        # Finding the min and max values to set the scale for the color bar
        vmax = np.max(np.abs(df_transposed.values))
        vmin = -vmax

        fig, ax = plt.subplots(figsize=(20, 10))
        cax = ax.imshow(df_transposed, cmap='seismic', vmin=vmin, vmax=vmax, aspect='auto')
        ax.set_title(f'Heatmap for {name}')
        ax.set_xlabel('Columns')
        ax.set_ylabel('Rows')

        # Adding vertical lines for nominal_response_tf and nominal_response_ti
        # Adjusting the positions considering the new focused index range starts from dist_response_ti
        ax.axvline(x=dist_response_tf - dist_response_ti, color='black', linestyle='-', linewidth=1)

        # Adding a colorbar to the plot
        fig.colorbar(cax, ax=ax)

    plt.show()

# Example usage
paths = [
    '/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/data/raw/ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR/robustness_gradients_analysis/0/0.02/-9/last_AnymalTerrain_ep_3800_rew_20.310041.pth/',
    '/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/data/raw/ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR/robustness_gradients_analysis/0/0.02/-9/last_AnymalTerrain_ep_4100_rew_20.68903.pth/'
]

data_frames = load_data(paths)
dist_response_ti, nom_response_tf, nom_response_ti, dist_response_tf = get_response_time_windows(raw_df1)
plot_data(data_frames, dist_response_ti, nom_response_tf, nom_response_ti, dist_response_tf)

print('hi')