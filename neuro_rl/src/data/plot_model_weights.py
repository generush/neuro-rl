import torch
import matplotlib.pyplot as plt
import numpy as np


def load_model(path):
    # Simulate loading a model and returning a dictionary representing the state_dict
    model =torch.load(path)
    state_dict = {key.replace('a2c_network.a_rnn.rnn.', ''): value for key, value in model['model'].items() if key.startswith('a2c_network.a_rnn.rnn')}

    model['parameters'] = {
        'bias_ih_l0': state_dict['bias_ih_l0'].cpu(),
        'bias_hh_l0': state_dict['bias_hh_l0'].cpu(),
        'weight_ih_l0': state_dict['weight_ih_l0'].cpu(),
        'weight_hh_l0': state_dict['weight_hh_l0'].cpu(),
        'weight_mlp': model['model']['a2c_network.actor_mlp.0.weight'][:,:176].cpu(),
        'bias_mlp': model['model']['a2c_network.actor_mlp.0.bias'].cpu()
    }

    return model

# Paths to your model files (use your actual paths)
model_paths = {
    "ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-1100": '/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/models/ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR/nn/last_AnymalTerrain_ep_1100_rew_14.392729.pth',
    "ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-2200": '/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/models/ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR/nn/last_AnymalTerrain_ep_2200_rew_19.53241.pth',
    "ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-3800": '/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/models/ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR/nn/last_AnymalTerrain_ep_3800_rew_20.310041.pth',
    "ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-4100": '/media/gene/fd75d0c8-aee1-476d-b68c-b17d9e0c1c14/code/NEURO/neuro-rl-sandbox/neuro-rl/neuro_rl/models/ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR/nn/last_AnymalTerrain_ep_4100_rew_20.68903.pth'
}

models = {name: load_model(path) for name, path in model_paths.items()}

weight_ih_l0_diff = models['ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-4100']['parameters']['weight_ih_l0'] - models['ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-3800']['parameters']['weight_ih_l0']
weight_hh_l0_diff =models['ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-4100']['parameters']['weight_hh_l0'] -  models['ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-3800']['parameters']['weight_hh_l0']

weight_mlp_diff = models['ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-4100']['parameters']['weight_mlp'] - models['ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-3800']['parameters']['weight_mlp']
bias_mlp_diff = models['ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-4100']['parameters']['bias_mlp'] - models['ANYMAL-1.0MASS-LSTM16-FRONTIERSDISTTERR-3800']['parameters']['bias_mlp']

import matplotlib.pyplot as plt

# Assuming weight_ih_l0_diff and weight_hh_l0_diff are your numpy arrays for plotting
# Adjust the limits for color scaling
vmax = max(abs(weight_ih_l0_diff.max()), abs(weight_ih_l0_diff.min()))
vmin = -vmax

# Plot for weight_ih_l0_diff
plt.figure(figsize=(10, 8))
plt.imshow(weight_ih_l0_diff, cmap='seismic', aspect='equal', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title('Difference in weight_ih_l0 between 1st and 4th Models')
for i in range(1, weight_ih_l0_diff.shape[0] // 128):
    plt.axhline(y=i*128, color='k', linestyle='-', linewidth=1)
for i in range(1, weight_ih_l0_diff.shape[1] // 128):
    plt.axvline(x=i*128, color='k', linestyle='-', linewidth=1)
plt.xlabel('Input Layer Neurons')
plt.ylabel('Hidden Layer Neurons (l0)')

# Adjust the limits for color scaling
vmax = max(abs(weight_hh_l0_diff.max()), abs(weight_hh_l0_diff.min()))
vmin = -vmax

# Plot for weight_hh_l0_diff
plt.figure(figsize=(10, 8))
plt.imshow(weight_hh_l0_diff, cmap='seismic', aspect='equal', vmin=vmin, vmax=vmax)
plt.colorbar()
plt.title('Difference in weight_hh_l0 between 1st and 4th Models')
for i in range(1, weight_hh_l0_diff.shape[0] // 128):
    plt.axhline(y=i*128, color='k', linestyle='-', linewidth=1)
for i in range(1, weight_hh_l0_diff.shape[1] // 128):
    plt.axvline(x=i*128, color='k', linestyle='-', linewidth=1)
plt.xlabel('Hidden Layer Neurons (t-1)')
plt.ylabel('Hidden Layer Neurons (t)')


# Plot the difference for weight_mlp
plt.figure(figsize=(10, 8))
plt.imshow(weight_mlp_diff, cmap='seismic', aspect='equal')
plt.colorbar()
plt.title('Difference in weight_mlp between 1st and 4th Models')
plt.xlabel('MLP Input Neurons')
plt.ylabel('MLP Output Neurons')

plt.show()

# # Plot the difference for weight_mlp
# plt.figure(figsize=(10, 8))
# plt.imshow(bias_mlp_diff, cmap='seismic', aspect='equal')
# plt.colorbar()
# plt.title('Difference in weight_mlp between 1st and 4th Models')
# plt.xlabel('MLP Input Neurons')
# plt.ylabel('MLP Output Neurons')
# plt.show()



print('hi')