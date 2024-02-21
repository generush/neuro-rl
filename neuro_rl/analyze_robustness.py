# https://plotly.com/python/3d-scatter-plots/
import pandas as pd
import matplotlib.pyplot as plt

DATA_PATH = '/home/gene/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data_CoRL/'
df = pd.read_csv(DATA_PATH + 'robustness_statistics.csv', index_col=0)
df = df.loc[:,'-4':'4']



fig, axs = plt.subplots(3, 2, figsize=(10, 6)) # Adjust the figure size as per your requirements

# Specify your colors for each row
colors = [['blue', 'lightblue'], ['green', 'lightgreen'], ['red', 'pink']]
patterns = ['', '//'] # Specify your patterns: '' for solid, '/' for hatched
alpha_values = [0.4, 0.3] # Specify transparency: 1 for opaque, less than 1 for transparent

for i in range(3):
    for j in range(2):
        # Assuming each subplot corresponds to two different models
        model1 = df.iloc[2*2*i + 2*j] # Adjust based on your DataFrame structure
        model2 = df.iloc[2*2*i + 2*j + 1] # Adjust based on your DataFrame structure

        # Plotting the bar chart for model1 and model2
        axs[i, j].bar(df.columns, model1, color=colors[i][0], edgecolor='black', hatch=patterns[0], alpha=alpha_values[0])
        axs[i, j].bar(df.columns, model2, color=colors[i][1], edgecolor='black', hatch=patterns[1], alpha=alpha_values[1])

        # Set the y limit
        axs[i, j].set_ylim([0, 1]) # Assuming percentage values range from 0 to 100

        # Adjust x-ticks frequency
        axs[i, j].set_xticks(axs[i, j].get_xticks()[::2])

        # Adding titles for each subplot
        # axs[i, j].set_title('Models {} and {}'.format(df.index[2*2*i + 2*j], df.index[2*2*i + 2*j + 1]))

# Adding common labels
# fig.text(0.5, 0.04, 'Conditions', ha='center')
# fig.text(0.04, 0.5, 'Success Percentage', va='center', rotation='vertical')

plt.tight_layout()

fig.savefig(DATA_PATH + 'robustness_statistics' + '.pdf', format='pdf', dpi=600, facecolor=fig.get_facecolor())

plt.show()



print('hi')

