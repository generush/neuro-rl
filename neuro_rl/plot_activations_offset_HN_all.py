import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt

import pickle as pk

DATA_PATH_ACTIVATION_HC = '/home/gene/Desktop/frontiers2023/activations_hc.csv'

DATA_PATH_SCL_PCA = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-09-27-15-49_u[1.0,1.0,1]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[10]/'

scl_hc = pk.load(open(DATA_PATH_SCL_PCA + 'A_LSTM_HC_SCL.pkl','rb'))
pca_hc = pk.load(open(DATA_PATH_SCL_PCA + 'A_LSTM_HC_PCA.pkl','rb'))

# hc = pk.load(open(DATA_PATH_ACTIVATION_HC))

hc = genfromtxt(DATA_PATH_ACTIVATION_HC, delimiter=',')
hc_mod = hc.copy()

hc_mod[:,128+6] = scl_hc.mean_[128+6]
hc_mod[:,128+13] = scl_hc.mean_[128+13]
hc_mod[:,128+18] = scl_hc.mean_[128+18]
hc_mod[:,128+54] = scl_hc.mean_[128+54]
hc_mod[:,128+60] = scl_hc.mean_[128+60]
hc_mod[:,128+73] = scl_hc.mean_[128+73]
hc_mod[:,128+94] = scl_hc.mean_[128+94]


hc_pc = pca_hc.transform(scl_hc.transform(hc))
hc_mod_pc = pca_hc.transform(scl_hc.transform(hc_mod))



plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# plot figures with speed colors and tangling colors
scatter1 = ax.plot(hc_pc[:,0], hc_pc[:,1], hc_pc[:,2], alpha=1, rasterized=True)

# Add labels and a legend
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_ylabel('PC 3')


ax.view_init(20, 55)



plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')


# plot figures with speed colors and tangling colors
scatter1 = ax.plot(hc_mod_pc[:,0], hc_mod_pc[:,1], hc_mod_pc[:,2], alpha=1, rasterized=True)

# Add labels and a legend
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_ylabel('PC 3')


ax.view_init(20, 55)
print('hello')




