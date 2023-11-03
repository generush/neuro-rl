import gpflow

from copy import deepcopy

import tensorflow as tf

import numpy as np

import dask.dataframe as dd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.cm import coolwarm



# lstm_model = torch.load('/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/runs/AnymalTerrain_2023-08-24_15-24-12/nn/last_AnymalTerrain_ep_3200_rew_20.145746.pth')
DATA_PATH = '/media/GENE_EXT4_2TB/code/NEURO/neuro-rl-sandbox/IsaacGymEnvs/isaacgymenvs/data/2023-08-27-16-41_u[0.4,1.0,14]_v[0.0,0.0,1]_r[0.0,0.0,1]_n[50]/'

# load DataFrame
df = dd.read_parquet(DATA_PATH + 'RAW_AND_PC_DATA' + '.parquet')
X = df['TIME'][:1000].compute().values.reshape(-1, 1).astype(np.float64) - 8
Y = df['A_LSTM_HC_PC_002'][:1000].compute().values.reshape(-1, 1).astype(np.float64)


def plot_kernel_samples(ax: Axes, kernel: gpflow.kernels.Kernel) -> None:
    X = np.zeros((0, 1))
    Y = np.zeros((0, 1))
    model = gpflow.models.GPR((X, Y), kernel=deepcopy(kernel))
    Xplot = np.linspace(-0.6, 0.6, 100)[:, None]
    tf.random.set_seed(20220903)
    n_samples = 3
    # predict_f_samples draws n_samples examples of the function f, and returns their values at Xplot.
    fs = model.predict_f_samples(Xplot, n_samples)
    ax.plot(Xplot, fs[:, :, 0].numpy().T, label=kernel.__class__.__name__)
    # ax.set_ylim(bottom=-2.0, top=2.0)
    ax.set_title("Example $f$s")


def plot_kernel_prediction(
    ax: Axes, kernel: gpflow.kernels.Kernel, *, optimise: bool = True
) -> None:
    # X = np.array([[-0.5], [0.0], [0.4], [0.5]])
    # Y = np.array([[1.0], [0.0], [0.6], [0.4]])
    model = gpflow.models.GPR(
        (X, Y), kernel=deepcopy(kernel), noise_variance=5e-1 # 1e-3
    )

    if optimise:
        gpflow.set_trainable(model.likelihood, False)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

    Xplot = np.linspace(-0.6, 0.6, 100)[:, None]

    f_mean, f_var = model.predict_f(Xplot, full_cov=False)
    f_lower = f_mean - 1.96 * np.sqrt(f_var)
    f_upper = f_mean + 1.96 * np.sqrt(f_var)

    ax.scatter(X, Y, color="black")
    (mean_line,) = ax.plot(Xplot, f_mean, "-", label=kernel.__class__.__name__)
    color = mean_line.get_color()
    ax.plot(Xplot, f_lower, color=color)
    ax.plot(Xplot, f_upper, color=color)
    ax.fill_between(
        Xplot[:, 0], f_lower[:, 0], f_upper[:, 0], color=color, alpha=0.1
    )
    # ax.set_ylim(bottom=-1.0, top=2.0)
    ax.set_title("Example data fit")


def plot_kernel(
    kernel: gpflow.kernels.Kernel, *, optimise: bool = True
) -> None:
    _, (samples_ax, prediction_ax) = plt.subplots(nrows=1, ncols=2)
    # plot_kernel_samples(samples_ax, kernel)
    plot_kernel_prediction(prediction_ax, kernel, optimise=optimise)


# _, ax = plt.subplots(nrows=1, ncols=1)

plot_kernel(
    gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(), period=0.48)
)

plt.show()

print('hi')