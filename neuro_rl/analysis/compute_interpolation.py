# https://plotly.com/python/3d-scatter-plots/

from typing import List

import numpy as np
import pandas as pd

from analysis.analyze_pca import compute_pca
from utils.io import export_pk

import sklearn.preprocessing
import sklearn.decomposition
import sklearn.manifold
import sklearn.metrics

import matplotlib
matplotlib.use('TkAgg')  # Replace 'TkAgg' with another backend if needed

import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from scipy.optimize import minimize

# https://datascience.stackexchange.com/questions/55066/how-to-export-pca-to-use-in-another-program
import pickle as pk


import os
from pdfCropMargins import crop

from constants.normalization_types import NormalizationType

def interpolate_dataframe(variable, time_periodic, fine_time):
    """Interpolate a given variable over a specified fine time grid."""
    spline = CubicSpline(time_periodic, np.append(variable, np.expand_dims(variable[0,:], axis=0), axis=0), bc_type='periodic')
    return spline(fine_time)

def compute_interpolation(df: pd.DataFrame):
    """Interpolate variables in a DataFrame over a specified fine time grid."""
    
    dt = df['TIME'][1] - df['TIME'][0]
    n_interp = 1000

    spd_cmd = df.loc[:, 'OBS_RAW_009_u_star']
    unique_speeds = np.unique(spd_cmd)
    
    interpolated_df = pd.DataFrame(columns=df.columns)

    for v in unique_speeds:

        filt_df = df[df['OBS_RAW_009_u_star'] == v]

        n = len(filt_df)
        time_periodic = np.arange(n + 1) * dt
        fine_time = np.linspace(0, n * dt, num=n_interp)

        single_speed_interpolated_np = interpolate_dataframe(filt_df.values, time_periodic, fine_time)
        single_speed_interpolated_df = pd.DataFrame(single_speed_interpolated_np, columns=df.columns)
        interpolated_df = pd.concat([interpolated_df, single_speed_interpolated_df], ignore_index=True)

    return interpolated_df