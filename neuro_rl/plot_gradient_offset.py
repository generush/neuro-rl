import matplotlib.pyplot as plt
import pandas as pd

def plot_csv_with_offset(csv_file, offset, start_index, stop_index, save_file):
    # Load the CSV file with 9 columns into a DataFrame
    df = pd.read_csv(csv_file, header=None)

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(10,4))

    # Plot each trace with the given offset within the specified index range
    for i in range(df.shape[1]):
        x_values = (df.index - df.index.min()) * 0.02  # Shift x-values to start at zero
        ax.plot(x_values[start_index:stop_index], df[i][start_index:stop_index] + offset * i, label=f'Trace {i + 1}')

    # Add labels and legend
    ax.set_xlabel('X-axis Label')
    ax.set_ylabel('Y-axis Label')
    ax.legend()

    # Save the figure as an SVG
    plt.savefig(save_file, format='svg')

    # Show the plot
    plt.show()

# Specify the CSV file, offset value, start index, and stop index
csv_file = '/home/gene/Desktop/frontiers2023/gradients.csv'
offset_value = -1  # Adjust the offset value as needed
start_index_value = 125  # Adjust the start index as needed
stop_index_value = 425  # Adjust the stop index as needed
save_file_name = 'gradients_timeplot.svg'

# Call the function to plot the data with the specified offset and custom x-labels
plot_csv_with_offset(csv_file, offset_value, start_index_value, stop_index_value, save_file_name)
print('hi')