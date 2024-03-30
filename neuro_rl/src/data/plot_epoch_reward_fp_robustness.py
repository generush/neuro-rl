import os
import pandas as pd
import matplotlib.pyplot as plt

import numpy as np

# Epochs
epochs4 = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 6000, 6700])

# Rewards
rewards4 = np.array([6.8, 10.1, 12.1, 12.5, 13.9, 14.3, 15.1, 13.2, 14.9, 11.7, 13.9, 15.3, 15.2, 14.1, 15.0, 14.4, 16.2, 15.8, 16.9, 17.2, 16.6, 16.3, 15.6, 16.1, 18.4, 18.9, 17.2, 17.2, 16.6, 18.1, 18.6, 17.1, 19.1, 18.5, 16.9, 20.0, 18.5, 18.8, 18.5, 19.0, 19.0, 18.0, 19.0, 19.8, 18.6, 18.9, 19.6, 17.8, 16.7, 20.1, 20.2])

# Number of false positives (no_fp)
no_fps4 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 4, 4, 4, 4, 4, 4, 4, 6, 6, 5, 6, 6, 6, 5, 5, 6, 5, 5, 5, 5, 4, 4, 5, 5, 4, 4, 5, 5, 5, 5, 5, 4, 4, 6, 5, 6, 6, 7, 7])

# Robustness
robustness4 = np.array([0, 0.28, 0.22, 0.24, 0.62, 0.71, 0.85, 0.79, 0.82, 0.88, 0.9, 0.86, 0.82, 0.83, 0.81, 0.84, 0.85, 0.87, 0.89, 0.89, 0.88, 0.96, 0.99, 0.92, 0.82, 0.78, 0.81, 0.8, 0.88, 0.88, 0.86, 0.96, 0.9, 0.96, 0.93, 0.9, 0.91, 0.98, 0.93, 0.99, 0.95, 0.97, 0.96, 0.98, 0.87, 0.92, 0.75, 0.8, 0.9, 0.72, 0.8])

# Epochs for LSTM16 dataset
epochs_lstm16 = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2500, 3000, 3500, 3700])

# Rewards for LSTM16 dataset
rewards_lstm16 = np.array([6.1, 8.4, 10.2, 11.5, 12.9, 13.6, 15.3, 12.5, 15.3, 10.7, 13.7, 13.7, 15.3, 15.2, 14.3, 15.5, 16.7, 15.3, 16.6, 16.6, 14.9, 17.8, 20.1])

# Number of false positives (no_fp) for LSTM16 dataset
no_fps_lstm16 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3])

# Robustness for LSTM16 dataset
robustness_lstm16 = np.array([0.25, 0.48, 0.52, 0.67, 0.69, 0.72, 0.80, 0.77, 0.89, 0.84, 0.85, 0.82, 0.87, 0.91, 0.84, 0.90, 0.96, 0.96, 1.00, 1.00, 1.00, 0.95, 0.99])

# Plot configuration for LSTM4 dataset
fig, axs = plt.subplots(3, 1, sharex=True)
fig.suptitle('LSTM4 Dataset')

# Epoch vs Reward for LSTM4
axs[0].plot(epochs4, rewards4, 'ko--', markersize=4, label='Reward')
axs[0].set_ylabel('Reward')
# axs[0].legend()

# Epoch vs No_FP for LSTM4
axs[1].plot(epochs4, no_fps4, 'rs--', markersize=4, label='No_FP')
axs[1].set_ylabel('Fixed Points')
axs[1].set_yticks([0,1,2,3,4,5,6,7])
# axs[1].legend()

# Epoch vs Robustness for LSTM4
axs[2].plot(epochs4, robustness4, 'b^--', markersize=4, label='Robustness')
axs[2].set_xlabel('Epoch')
axs[2].set_ylabel('Robustness')
# axs[2].legend()

# Plot configuration for LSTM16 dataset
fig2, axs2 = plt.subplots(3, 1, sharex=True)
fig2.suptitle('LSTM16')

# Epoch vs Reward for LSTM16
axs2[0].plot(epochs_lstm16, rewards_lstm16, 'ko--', markersize=4, label='Reward')
axs2[0].set_ylabel('Reward')
# axs2[0].legend()

# Epoch vs No_FP for LSTM16
axs2[1].plot(epochs_lstm16, no_fps_lstm16, 'rs--', markersize=4, label='No_FP')
axs2[1].set_ylabel('Fixed Points')
axs2[1].set_yticks([0,1,2,3])
# axs2[1].legend()    

# Epoch vs Robustness for LSTM16
axs2[2].plot(epochs_lstm16, robustness_lstm16, 'b^--', markersize=4, label='Robustness')
axs2[2].set_xlabel('Epoch')
axs2[2].set_ylabel('Robustness')
# axs2[2].legend()

# Adjust the layout
plt.tight_layout()
plt.subplots_adjust(top=0.93, hspace=0.2)

import matplotlib.pyplot as plt
from scipy import stats

# Function to perform linear regression and plot the results
def plot_regression(ax, x, y, color):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # Create a line of best fit
    line = slope * x + intercept
    # Plot the original scatter plot
    ax.scatter(x, y, color=color)
    # Plot the line of best fit
    ax.plot(x, line, color='gray', linestyle='-', linewidth=1, label=f'Fit Line ($R^2={r_value**2:.2f}$)')
    # Add the legend to the plot
    ax.legend()

# Initialize the subplot with 2 rows and 4 columns
fig, axs = plt.subplots(2, 3, figsize=(9, 6))
fig.suptitle('Scatter Plots with Least Squares Fit for LSTM4 and LSTM16 Datasets')

# Plot for LSTM4 dataset: Reward vs Robustness
plot_regression(axs[0, 0], rewards4, robustness4, 'orange')
axs[0, 0].set_xlabel('Reward')
axs[0, 0].set_ylabel('Robustness')

# Plot for LSTM4 dataset: Fixed Points vs Reward
plot_regression(axs[0, 1], no_fps4, rewards4, 'blue')
axs[0, 1].set_xlabel('Fixed Points')
axs[0, 1].set_ylabel('Reward')

# Plot for LSTM4 dataset: Fixed Points vs Robustness
plot_regression(axs[0, 2], no_fps4, robustness4, 'green')
axs[0, 2].set_xlabel('Fixed Points')
axs[0, 2].set_ylabel('Robustness')

# Plot for LSTM16 dataset: Reward vs Robustness
plot_regression(axs[1, 0], rewards_lstm16, robustness_lstm16, 'magenta')
axs[1, 0].set_xlabel('Reward')
axs[1, 0].set_ylabel('Robustness')

# Plot for LSTM16 dataset: Fixed Points vs Reward
plot_regression(axs[1, 1], no_fps_lstm16, rewards_lstm16, 'red')
axs[1, 1].set_xlabel('Fixed Points')
axs[1, 1].set_ylabel('Reward')

# Plot for LSTM16 dataset: Fixed Points vs Robustness
plot_regression(axs[1, 2], no_fps_lstm16, robustness_lstm16, 'purple')
axs[1, 2].set_xlabel('Fixed Points')
axs[1, 2].set_ylabel('Robustness')

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()



print('hi')