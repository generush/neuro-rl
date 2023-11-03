import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from scipy import interpolate


# Define the base directory
base_dir = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/'

# data_subdirectories = [
#     '2023-09-19-14-26_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-0.0BW_NODIST_NOTERR/',
#     '2023-09-19-14-25_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-0.5BW_NODIST_NOTERR/',
#     '2023-09-19-14-24_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-2.0BW_NODIST_NOTERR/',
#     '2023-09-19-14-23_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-3.5BW_NODIST_NOTERR/',
# ]

# data_subdirectories = [
#     '2023-09-19-14-18_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-0.0BW_DIST_NOTERR/',
#     '2023-09-19-14-17_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-0.5BW_DIST_NOTERR/',
#     '2023-09-19-14-16_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-2.0BW_DIST_NOTERR/',
#     '2023-09-19-14-15_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-3.5BW_DIST_NOTERR/',
# ]

# data_subdirectories = [
#     '2023-09-19-14-02_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-0.0BW_NODIST_TERR/',
#     '2023-09-19-13-49_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-0.5BW_NODIST_TERR/',
#     '2023-09-19-13-47_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-2.0BW_NODIST_TERR/',
#     '2023-09-19-13-46_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-3.5BW_NODIST_TERR/',
# ]

# data_subdirectories = [
#     '2023-09-19-13-19_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-0.0BW_DIST_TERR/',
#     '2023-09-19-13-17_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-0.5BW_DIST_TERR/',
#     '2023-09-19-13-15_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-2.0BW_DIST_TERR/',
#     '2023-09-19-13-12_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-3.5BW_DIST_TERR/',
# ]

# data_subdirectories = [
#     '2023-09-19-14-24_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-2.0BW_NODIST_NOTERR/',
#     '2023-09-19-14-16_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-2.0BW_DIST_NOTERR/',
#     '2023-09-19-13-47_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-2.0BW_NODIST_TERR/',
#     '2023-09-19-13-15_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-2.0BW_DIST_TERR/',
# ]

data_subdirectories = [
    '2023-09-19-14-23_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-3.5BW_NODIST_NOTERR/',
    '2023-09-19-14-15_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-3.5BW_DIST_NOTERR/',
    '2023-09-19-13-46_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-3.5BW_NODIST_TERR/',
    '2023-09-19-13-12_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-3.5BW_DIST_TERR/',
]

# Load your dataframes and store them in a dictionary
dataframes = {}
for i, subdirectory in enumerate(data_subdirectories, start=1):
    dataframes[f'DataFrame{i}'] = pd.read_csv(base_dir + subdirectory + 'RAW_DATA.csv', index_col=0)


# Loop through the dataframes and add FT_CONTACT_001 based on FT_FORCE_RAW_000
for name, dataframe in dataframes.items():
    dataframes[name]['FT_CONTACT_LF'] = np.where(dataframe['FT_FORCE_RAW_000'] > 0, 4, np.nan)
    dataframes[name]['FT_CONTACT_LH'] = np.where(dataframe['FT_FORCE_RAW_001'] > 0, 1, np.nan)
    dataframes[name]['FT_CONTACT_RF'] = np.where(dataframe['FT_FORCE_RAW_002'] > 0, 3, np.nan)
    dataframes[name]['FT_CONTACT_RH'] = np.where(dataframe['FT_FORCE_RAW_003'] > 0, 2, np.nan)

# Define the TIME points where you want to add vertical lines (adjust as needed)
# vertical_lines = [-0.8, -0.4, 0.0, 0.4, 0.8, 1.2]  # Example: Add vertical lines at times 4.5 and 5.0

# Define the TIME range for the light gray translucent region (adjust as needed)
translucent_time_range = (0.0, 0.08)  # Example: Add light gray translucent region from 4.7 to 5.2

line_thickness = [
    2, 
    2, 
    2, 
    2
]

line_styles = [
    '-',
    '-',
    '-',
    '-',
]

legend_names = [
    'Baseline',
    'Dist',
    'Terr',
    'DistTerr',
]

# Create a figure and axes for subplots
fig, axs = plt.subplots(8, 1, figsize=(10, 12), sharex=True)  # Share the x-axis for subplots

# Define colors for plotting
colors = ['k', 'b', 'g', 'r']

fields_to_plot = [
    'OBS_RAW_001_v',
    'ACT_RAW_009_RH_HAA',
    'OBS_RAW_021_dof_pos_10',
    'OBS_RAW_007_theta',
]

# Define nicknames for the fields (corresponding to the order in fields_to_plot)
field_nicknames = [
    'v [m/s]',
    'RH \nHip Torque \nCommand \n[Nm]',
    'RH \nHip Position \n[deg]',
    r'$\theta$ [deg/s]',
]

# Initialize a list to store done_index for each dataframe
done_indices = []

# Loop through the DataFrame names
for name, df in dataframes.items():
    if (dataframes[name]['DONE'] == 1).sum() > 0:
        done = (dataframes[name]['DONE'] == 1)
        done_index = done[done].index[0] + 1
    else:
        done_index = len(dataframes[name]) + 1
    done_indices.append(done_index)

    perturb = (dataframes[name]['PERTURB_BEGIN'] == 1)
    perturb_index = perturb[perturb].index[0]
    dataframes[name]['TIME'] = dataframes[name]['TIME'] - dataframes[name]['TIME'].iloc[perturb_index] + 0.02


    # Conver to deg
    dataframes[name]['OBS_RAW_021_dof_pos_10'] = 180 / np.pi * np.arccos(dataframes[name]['OBS_RAW_021_dof_pos_10'])

    # Compute OBS_RAW_007_theta
    dataframes[name]['OBS_RAW_007_theta'] = 180 / np.pi * np.arccos(dataframes[name]['OBS_RAW_007_theta_proj'])


# Loop through the dataframes and plot data on subplots
for i, (name, df) in enumerate(dataframes.items()):
    done_index = done_indices[i]
    for j, col in enumerate(fields_to_plot):
        axs[j].plot(df['TIME'][:done_index], df[col][:done_index], label=f'{legend_names[i]}', color=colors[i], linewidth=line_thickness[i], linestyle=line_styles[i])

        # Set y-axis label to the field name
        axs[j].set_ylabel(field_nicknames[j])

    # # Add vertical dashed lines at specified TIME points
    # for line_time in vertical_lines:
    #     for ax in axs:
    #         ax.axvline(x=line_time, color='black', linestyle='--')

    # Add light gray translucent region
    for ax in axs:
        ax.axvspan(translucent_time_range[0], translucent_time_range[1], facecolor='lightgray', alpha=0.5)

contact_force_fields = [
    'FT_CONTACT_LF', 
    'FT_CONTACT_RF', 
    'FT_CONTACT_RH', 
    'FT_CONTACT_LH'
]

# # Define nicknames for the fields (corresponding to the order in fields_to_plot)
# contact_nicknames = [
#     ''''''
# ]

# Loop through the dataframes and plot data on subplots
for i, (name, df) in enumerate(dataframes.items()):
    for j, col in enumerate(contact_force_fields):
        axs[i+4].plot(df['TIME'][:done_index], df[col][:done_index], label=f'{name} {col}', color=colors[i], linewidth=line_thickness[i], linestyle=line_styles[i])

    # # Add vertical dashed lines at specified TIME points
    # for line_time in vertical_lines:
    #     for ax in axs:
    #         ax.axvline(x=line_time, color='black', linestyle='--')

    # Add light gray translucent region
    for ax in axs:
        ax.axvspan(translucent_time_range[0], translucent_time_range[1], facecolor='lightgray', alpha=0.5)

# # Adjust subplot spacing
# plt.tight_layout()

# Customize legend and labels
for ax in axs:
    ax.legend()
    ax.set_xlim(-0.8, 1.2)
    ax.set_xlabel('Time')

# Save the figure as an SVG file
fig.savefig('perturb_response_biomechanics.svg', format='svg')

# Show the plot
plt.show()




# # Show the plot
# plt.show()



# # Create a meshgrid for x and y values
# df.OBS_RAW_001_v
# df.OBS_RAW_007_theta_proj = 
# df.ACT_RAW_000_LF_HAA
# df.ACT_RAW_001_LF_HFE
# df.ACT_RAW_002_LF_KFE
# df.ACT_RAW_003_LH_HAA
# df.ACT_RAW_004_LH_HFE
# df.ACT_RAW_005_LH_KFE
# df.ACT_RAW_006_RF_HAA
# df.ACT_RAW_007_RF_HFE
# df.ACT_RAW_008_RF_KFE
# df.ACT_RAW_009_RH_HAA
# df.ACT_RAW_010_RH_HFE
# df.ACT_RAW_011_RH_KFE

# df.FT_FORCE_RAW_000
# df.FT_FORCE_RAW_001
# df.FT_FORCE_RAW_002
# df.FT_FORCE_RAW_003

# df.TIME

# Show the plot

print('hi')