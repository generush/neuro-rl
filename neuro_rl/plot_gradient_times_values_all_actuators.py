import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt

from matplotlib.colors import LinearSegmentedColormap

import pickle as pk

action_cn = np.zeros([500,128,12])
action_cn_grad = np.zeros([500,128,12])
action_cn[:,:,0] = genfromtxt('/home/gene/Pictures/action_00_LF_hip/cn_in.csv', delimiter=',')
action_cn[:,:,1] = genfromtxt('/home/gene/Pictures/action_01_LF_shoulder/cn_in.csv', delimiter=',')
action_cn[:,:,2] = genfromtxt('/home/gene/Pictures/action_02_LF_knee/cn_in.csv', delimiter=',')
action_cn[:,:,3] = genfromtxt('/home/gene/Pictures/action_03_LH_hip/cn_in.csv', delimiter=',')
action_cn[:,:,4] = genfromtxt('/home/gene/Pictures/action_04_LH_shoulder/cn_in.csv', delimiter=',')
action_cn[:,:,5] = genfromtxt('/home/gene/Pictures/action_05_LH_knee/cn_in.csv', delimiter=',')
action_cn[:,:,6] = genfromtxt('/home/gene/Pictures/action_06_RF_hip/cn_in.csv', delimiter=',')
action_cn[:,:,7] = genfromtxt('/home/gene/Pictures/action_07_RF_shoulder/cn_in.csv', delimiter=',')
action_cn[:,:,8] = genfromtxt('/home/gene/Pictures/action_08_RF_knee/cn_in.csv', delimiter=',')
action_cn[:,:,9] = genfromtxt('/home/gene/Pictures/action_09_RH_hip/cn_in.csv', delimiter=',')
action_cn[:,:,10] = genfromtxt('/home/gene/Pictures/action_10_RH_shoulder/cn_in.csv', delimiter=',')
action_cn[:,:,11] = genfromtxt('/home/gene/Pictures/action_11_RH_knee/cn_in.csv', delimiter=',')

action_cn_grad[:,:,0] = genfromtxt('/home/gene/Pictures/action_00_LF_hip/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,1] = genfromtxt('/home/gene/Pictures/action_01_LF_shoulder/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,2] = genfromtxt('/home/gene/Pictures/action_02_LF_knee/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,3] = genfromtxt('/home/gene/Pictures/action_03_LH_hip/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,4] = genfromtxt('/home/gene/Pictures/action_04_LH_shoulder/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,5] = genfromtxt('/home/gene/Pictures/action_05_LH_knee/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,6] = genfromtxt('/home/gene/Pictures/action_06_RF_hip/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,7] = genfromtxt('/home/gene/Pictures/action_07_RF_shoulder/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,8] = genfromtxt('/home/gene/Pictures/action_08_RF_knee/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,9] = genfromtxt('/home/gene/Pictures/action_09_RH_hip/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,10] = genfromtxt('/home/gene/Pictures/action_10_RH_shoulder/cn_in_grad.csv', delimiter=',')
action_cn_grad[:,:,11] = genfromtxt('/home/gene/Pictures/action_11_RH_knee/cn_in_grad.csv', delimiter=',')



action_cn_grad_times_input_values = action_cn * action_cn_grad





baseline = np.mean(action_cn_grad_times_input_values[230:250,:,:],axis=0)
disturbed = np.mean(action_cn_grad_times_input_values[250:270,:,:],axis=0)
diff = disturbed - baseline

# diff[0,0] = -100

# Create a custom colormap
colors = [(1, 0, 0), (1, 1, 1), (0, 0, 1)]  # Red, White, Blue
n_bins = 200  # Number of bins
cmap_name = "custom_colormap"
custom_cmap = LinearSegmentedColormap.from_list(cmap_name, colors, N=n_bins)


vmin = diff.min()
vmax = diff.max()
vvmax = max([abs(diff.min()), (diff.max())])

# Create a colorgrid plot
plt.figure(figsize=(18, 6))
plt.imshow(diff.transpose(), aspect='auto', cmap=custom_cmap, vmin=-vvmax, vmax=vvmax)
plt.gca().invert_yaxis()

# Add labels and colorbar
plt.colorbar(label='Value')
plt.xlabel('Neuron')
plt.ylabel('Actuator')

plt.title('Colorgrid Plot of a 128x12 Dataset')
plt.show()

print('hellow')


