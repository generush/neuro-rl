import matplotlib.pyplot as plt
import pandas as pd

def plot_csv_with_offset(csv_file, offset, start_index, stop_index, baseline_start_index, baseline_stop_index, alpha=1.0, save_file=None):
    # Load the CSV file with 9 columns into a DataFrame
    df = pd.read_csv(csv_file, header=None)

    # Create a Matplotlib figure and axes
    fig, ax = plt.subplots(figsize=(10, 4))

    # Define a list of colors to use
    colors = 256 * ['b']
    
    # Plot each trace with the given offset and alpha within the specified index range
    for i in range(df.shape[1]):
        
        x_values = (df.index - df.index.min()) * 0.02  # Shift x-values to start at zero
        ax.plot(x_values[start_index:stop_index], df[i][start_index:stop_index] + offset * i,
                label=f'Trace {i + 1}', color=colors[i], linewidth=0.5)
        # ax.plot(x_values[baseline_start_index:baseline_stop_index] + (start_index_value-baseline_start_index_value)/50, df[i][baseline_start_index:baseline_stop_index] + offset * i,
        #         label=f'Trace {i + 1}', linestyle='dashed', alpha=alpha, color=colors[i])

    # Add labels and legend
    ax.set_xlabel('X-axis Label')
    ax.set_ylabel('Y-axis Label')
    # ax.legend()

    # Save the figure as an SVG if a save file is provided
    if save_file:
        plt.savefig(save_file, format='svg')

    # Show the plot
    plt.show()

# Specify the CSV file, offset value, start index, and stop index for the second set of 10 traces
csv_file = '/home/gene/Desktop/frontiers2023/activations_cn.csv'  # Change this to your overlay CSV file
offset_value = -10  # Adjust the overlay offset value as needed
start_index_value = 230  # Adjust the overlay start index as needed
stop_index_value = 350  # Adjust the overlay stop index as needed
baseline_start_index_value = 106  # Adjust the overlay start index as needed
baseline_stop_index_value = 226  # Adjust the overlay stop index as needed

# Call the function to plot the second set of 10 traces with dashed lines, alpha=0.5, and a different time window
plot_csv_with_offset(csv_file, offset_value, start_index_value, stop_index_value, baseline_start_index_value, baseline_stop_index_value, alpha=0.5)
