import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define the file paths for your data
DATA_PATH_0 = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-09-17-15-08_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-0.0BW_DIST-TERR/'
DATA_PATH_1 = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-09-16-12-00_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-0.5BW_DIST_TERR/'
DATA_PATH_2 = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-09-16-11-58_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-2.0BW_DIST_TERR/'
DATA_PATH_3 = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-09-16-12-01_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[1]_-3.5BW_DIST_TERR/'

# Load your four dataframes and store them in a dictionary
dataframes = {
    'DataFrame1': pd.read_csv(DATA_PATH_0 + 'RAW_DATA.csv', index_col=0),
    'DataFrame2': pd.read_csv(DATA_PATH_1 + 'RAW_DATA.csv', index_col=0),
    'DataFrame3': pd.read_csv(DATA_PATH_2 + 'RAW_DATA.csv', index_col=0),
    'DataFrame4': pd.read_csv(DATA_PATH_3 + 'RAW_DATA.csv', index_col=0),
}

# Define the shifts for each dataframe (adjust as needed)
shifts = {
    'DataFrame1': 0,
    'DataFrame2': 0.04,
    'DataFrame3': 0.04,
    'DataFrame4': 0,
}


# Define the common time window for all dataframes (adjust as needed)
common_time_window = (4.2, 6.6)

# Define the TIME points where you want to add vertical lines (adjust as needed)
vertical_lines = [4.6, 5.0, 5.4, 5.8, 6.2]

# Define the TIME range for the light gray translucent region (adjust as needed)
translucent_time_range = (5.0, 5.1)

# Define the line thickness for each plot (adjust as needed)
line_thickness = [3, 1, 1, 1]

# Function to shift the 'TIME' column in a dataframe by N steps
def shift_dataframe(df, shift_steps):
    df['TIME'] = df['TIME'] + shift_steps
    return df

# Function to filter dataframe based on time window
def filter_dataframe_by_time(df, start_time, end_time):
    return df[(df['TIME'] >= start_time) & (df['TIME'] <= end_time)]

# Compute contact times for each foot within the common time window
def compute_contact_times(force_data, time_data, start_time, end_time):
    time_index = (time_data >= start_time) & (time_data <= end_time)
    contact_times = np.where(force_data[time_index] > 0, 1, np.nan)
    return contact_times

# Assuming you have the 'dataframes' containing the relevant data
force_LF = dataframes['DataFrame1']['FT_FORCE_RAW_000']
force_LH = dataframes['DataFrame1']['FT_FORCE_RAW_001']
force_RF = dataframes['DataFrame1']['FT_FORCE_RAW_002']
force_RH = dataframes['DataFrame1']['FT_FORCE_RAW_003']
time_data = dataframes['DataFrame1']['TIME']

# Shift each dataframe by its specified number of steps
for name, df in dataframes.items():
    if name in shifts:
        dataframes[name] = shift_dataframe(df, shifts[name])

# Filter each dataframe based on the common time window
for name, df in dataframes.items():
    df = filter_dataframe_by_time(df, *common_time_window)
    dataframes[name] = df

# Compute contact times for each foot within the common time window
contact_LF = compute_contact_times(force_LF, time_data, *common_time_window)
contact_LH = compute_contact_times(force_LH, time_data, *common_time_window)
contact_RF = compute_contact_times(force_RF, time_data, *common_time_window)
contact_RH = compute_contact_times(force_RH, time_data, *common_time_window)

# Create a new subplot for foot contact
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

# Create a time index for the specified time window
time_index = time_data[(time_data >= common_time_window[0]) & (time_data <= common_time_window[1])]

# Plot the foot contact data within the common time window
axs[0].plot(time_index, contact_LF, label='LF Contact Time', color='b', linewidth=2)
axs[1].plot(time_index, contact_LH, label='LH Contact Time', color='g', linewidth=2)
axs[2].plot(time_index, contact_RF, label='RF Contact Time', color='r', linewidth=2)
axs[3].plot(time_index, contact_RH, label='RH Contact Time', color='y', linewidth=2)

# Add vertical dashed lines at specified TIME points
for line_time in vertical_lines:
    for ax in axs:
        ax.axvline(x=line_time, color='black', linestyle='--')

# Add light gray translucent region
for ax in axs:
    ax.axvspan(translucent_time_range[0], translucent_time_range[1], facecolor='lightgray', alpha=0.5)

# Adjust subplot spacing
plt.tight_layout()

# Customize legend and labels
for ax in axs:
    ax.legend()
    ax.set_ylabel('Contact Time')

# Show the plot
plt.xlabel('Time')
plt.show()

print('hi')