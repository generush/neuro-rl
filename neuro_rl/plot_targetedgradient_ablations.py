import matplotlib.pyplot as plt
import numpy as np

# Define x-axis values
x_values = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 32, 64, 96, 128]  # X-axis values from 0 to 128 (integer)
x_values2 = [0, 32, 64, 96, 128]  # X-axis values from 0 to 128 (integer)

# Manually specify data points for four datasets, each containing 6 data points
data1 = [400, 395, 397, 242, 168, 270, 241, 148, 188, 117, 23, 46, 1, 0, 0, 0, 0, 0, 0, 0, 0]  # Recovery rate for dataset 1
data2 = [400, 400, 400, 399, 399, 400, 400, 400, 399, 400, 400, 400, 400, 400, 400, 400, 400, 400, 382, 384, 219]  # Recovery rate for dataset 2
data3 = [400, 400, 400, 398, 391, 392, 388, 371, 364, 233, 234, 295, 299, 297, 306, 249, 277, 276, 51, 3, 0]  # Recovery rate for dataset 3

# Manually specify data points for three additional datasets, each containing 5 data points
data4 = [400, 59, 0, 0, 0]  # Recovery rate for dataset 4
data5 = [400, 394, 353, 283, 182]  # Recovery rate for dataset 5
data6 = [400, 345, 179, 57, 8]  # Recovery rate for dataset 6

data1_normalized = np.array(data1) / 400
data2_normalized = np.array(data2) / 400
data3_normalized = np.array(data3) / 400
data4_normalized = np.array(data4) / 400
data5_normalized = np.array(data5) / 400
data6_normalized = np.array(data6) / 400

percentage_ablated = [x_val / 128 * 100 for x_val in x_values]

# Create a figure and a set of subplots
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Scatter plot and line plot for dataset 1
plt.plot(x_values, data1_normalized, label='Targeted Ablations: Hidden Neurons (Post-RNN)', c='b', linestyle='-', alpha=1, marker='o')
plt.plot(x_values, data2_normalized, label='Targeted Ablations: Hidden Neurons (Pre-RNN)', c='g', linestyle='-', alpha=1, marker='s')
plt.plot(x_values, data3_normalized, label='Targeted Ablations: Cell Neurons (Pre-RNN)', c='r', linestyle='-', alpha=1, marker='^')
plt.plot(x_values2, data4_normalized, label='Random Ablations: Hidden Neurons (Post-RNN)', c='b', linestyle='--', alpha=1, marker='v')
plt.plot(x_values2, data5_normalized, label='Random Ablations: Hidden Neurons (Pre-RNN)', c='g', linestyle='--', alpha=1, marker='d')
plt.plot(x_values2, data6_normalized, label='Random Ablations: Cell Neurons (Pre-RNN)', c='r', linestyle='--', alpha=1, marker='x')

# Add labels and legend
plt.xlabel('Number of Neurons Ablated')
plt.ylabel('Recovery Rate')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()







