import matplotlib.pyplot as plt
import numpy as np


# Define x-axis values
x_values = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,32,48,64,80,96,112,128]  # X-axis values from 0 to 128 (integer)

# Manually specify data points for four datasets, each containing 6 data points
data1 = [400,399,400,397,379,361,324,196,111,40,8,9,0,0,1,2,0,0,0,0,0,0,0,0]  # Recovery rate for dataset 1
data2 = [400,399,399,400,399,400,400,400,400,400,398,400,400,400,400,399,400,400,400,383,374,341,258,211]  # Recovery rate for dataset 2
data3 = [400,400,397,398,391,387,396,346,339,318,337,349,343,340,322,312,303,301,195,109,105,42,0,14]  # Recovery rate for dataset 3

# Manually specify data points for three additional datasets, each containing 5 data points
data4 = [400,399,395,391,388,383,368,350,361,347,337,328,310,291,287,284,256,53,6,0,0,0,0,0]  # Recovery rate for dataset 4
data5 = [400,400,399,400,396,400,399,400,399,398,400,398,398,400,400,399,399,392,381,358,331,308,253,243]  # Recovery rate for dataset 5
data6 = [400,399,400,399,398,397,394,396,397,395,393,395,390,399,390,392,386,316,247,158,94,66,14,4]  # Recovery rate for dataset 6






# # Define x-axis values
# x_values = [0,16,32,48,64,80,96,112,128]  # X-axis values from 0 to 128 (integer)

# # Manually specify data points for four datasets, each containing 6 data points
# data1 = [400,0,0,0,0,0,0,0,0]  # Recovery rate for dataset 1
# data2 = [400,400,400,400,383,374,341,258,211]  # Recovery rate for dataset 2
# data3 = [400,303,301,195,109,105,42,0,14]  # Recovery rate for dataset 3

# # Manually specify data points for three additional datasets, each containing 5 data points
# data4 = [400,256,53,6,0,0,0,0,0]  # Recovery rate for dataset 4
# data5 = [400,399,392,381,358,331,308,253,243]  # Recovery rate for dataset 5
# data6 = [400,386,316,247,158,94,66,14,4]  # Recovery rate for dataset 6






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
plt.plot(x_values, data4_normalized, label='Random Ablations: Hidden Neurons (Post-RNN)', c='b', linestyle='--', alpha=1, marker='v')
plt.plot(x_values, data5_normalized, label='Random Ablations: Hidden Neurons (Pre-RNN)', c='g', linestyle='--', alpha=1, marker='d')
plt.plot(x_values, data6_normalized, label='Random Ablations: Cell Neurons (Pre-RNN)', c='r', linestyle='--', alpha=1, marker='x')

# Add labels and legend
plt.xticks([0,16,32,48,64,80,96,112,128])
plt.xlabel('Number of Neurons Ablated')
plt.ylabel('Recovery Rate')
plt.legend()

# Display the plot
plt.grid(True)
plt.show()







