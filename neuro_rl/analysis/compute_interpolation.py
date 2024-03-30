# https://plotly.com/python/3d-scatter-plots/


import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('TkAgg')  # Replace 'TkAgg' with another backend if needed


from scipy.interpolate import CubicSpline

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program

def interpolate_cyclic_dataframe(variable, time_periodic, fine_time):
    """Interpolate a given variable over a specified fine time grid."""
    spline = CubicSpline(time_periodic, np.append(variable, np.expand_dims(variable[0,:], axis=0), axis=0), bc_type='periodic')
    return spline(fine_time)

def interpolate_dataframe(variable, time, fine_time):
    """Interpolate a given variable over a specified fine time grid for non-cyclic data."""
    # No need to append the first row to the end for non-cyclic interpolation
    spline = CubicSpline(time, variable)
    return spline(fine_time)


def compute_cyclic_interpolation(df: pd.DataFrame):
    """Interpolate variables in a DataFrame over a specified fine time grid."""
    
    dt = df['TIME_RAW'][1] - df['TIME_RAW'][0]
    n_interp = 1000

    spd_cmd = df.loc[:, 'OBS_RAW_009_u_star']
    unique_speeds = np.unique(spd_cmd)
    
    interpolated_df = pd.DataFrame(columns=df.columns)

    for v in unique_speeds:

        filt_df = df[df['OBS_RAW_009_u_star'] == v]

        n = len(filt_df)
        time_periodic = np.arange(n + 1) * dt
        fine_time = np.linspace(0, n * dt, num=n_interp)

        single_speed_interpolated_np = interpolate_cyclic_dataframe(filt_df.values, time_periodic, fine_time)
        single_speed_interpolated_df = pd.DataFrame(single_speed_interpolated_np, columns=df.columns)
        interpolated_df = pd.concat([interpolated_df, single_speed_interpolated_df], ignore_index=True)

    return interpolated_df

def compute_interpolation(df: pd.DataFrame):
    """Interpolate variables in a DataFrame over a specified fine time grid."""
    
    dt = df['TIME_RAW'][1] - df['TIME_RAW'][0]
    n_interp = 5000

    run_id_data = df.loc[:, 'RUN_ID']
    run_ids = np.unique(run_id_data)
    
    interpolated_df = pd.DataFrame(columns=df.columns)

    for run_id in run_ids:

        filt_df = df[df['RUN_ID'] == run_id]

        n = len(filt_df)
        time = np.arange(n) * dt
        fine_time = np.linspace(0, n * dt, num=n_interp)

        single_speed_interpolated_np = interpolate_dataframe(filt_df.values, time, fine_time)
        single_speed_interpolated_df = pd.DataFrame(single_speed_interpolated_np, columns=df.columns)
        interpolated_df = pd.concat([interpolated_df, single_speed_interpolated_df], ignore_index=True)

    return interpolated_df