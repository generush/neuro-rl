import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

# Define x-axis values
x_values = [0, 32, 64, 96, 128] # X-axis values from 0 to 128 (integer)

# Manually specify data points for four datasets, each containing 6 data points
data1 = [400, 59, 0, 0, 0]  # Recovery rate for dataset 1
data2 = [400, 394, 353, 283, 182]  # Recovery rate for dataset 2
data3 = [400, 345, 179, 57, 8]  # Recovery rate for dataset 3

percentage_ablated = [x_val / 128 * 100 for x_val in x_values]

# Create a scatter plot for each dataset
plt.figure(figsize=(10, 6))  # Adjust the figure size as needed

# Scatter plot for dataset 1
plt.plot(x_values, data1, label='Post-RNN Hidden Neurons', c='b', marker='o')

# Scatter plot for dataset 2
plt.plot(x_values, data2, label='Pre-RNN Hidden Neurons', c='g', marker='s')

# Scatter plot for dataset 3
plt.plot(x_values, data3, label='Pre-RNN Cell Neurons', c='r', marker='^')

# Add labels and legend
plt.xlabel('Number of Neurons Ablated')
plt.ylabel('Recovery Rate')
# plt.title('Scatter Plot with 4 Different Datasets (Each with 6 Data Points)')
plt.legend()


# Set y-axis limits
# plt.ylim(0, 100)  # Limit the y-axis to a percentage range (0-100)

# # Create a secondary (top) x-axis with custom label
# top_ax = plt.twiny()
# top_ax.set_xlim(0, 100)  # Set the limits of the top x-axis to match the percentage range
# top_ax.set_xlabel('Percentage of Neurons Ablated')
# custom_ticks = [0, 25, 50, 75, 100]  # Custom tick positions
# custom_tick_positions = [percentage_ablated[x_values.index(tick)] for tick in custom_ticks]  # Corresponding positions
# top_ax.set_xticks(custom_tick_positions)
# top_ax.set_xticklabels([f'{tick}%' for tick in custom_ticks])  # Display the labels as percentages



# Display the plot
# plt.grid(True)
plt.show()

