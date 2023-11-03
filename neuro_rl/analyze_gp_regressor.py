"""
==========================================================
Comparison of kernel ridge and Gaussian process regression
==========================================================

This example illustrates differences between a kernel ridge regression and a
Gaussian process regression.

Both kernel ridge regression and Gaussian process regression are using a
so-called "kernel trick" to make their models expressive enough to fit
the training data. However, the machine learning problems solved by the two
methods are drastically different.

Kernel ridge regression will find the target function that minimizes a loss
function (the mean squared error).

Instead of finding a single target function, the Gaussian process regression
employs a probabilistic approach : a Gaussian posterior distribution over
target functions is defined based on the Bayes' theorem, Thus prior
probabilities on target functions are being combined with a likelihood function
defined by the observed training data to provide estimates of the posterior
distributions.

We will illustrate these differences with an example and we will also focus on
tuning the kernel hyperparameters.
"""

# Authors: Jan Hendrik Metzen <jhm@informatik.uni-bremen.de>
#          Guillaume Lemaitre <g.lemaitre58@gmail.com>
# License: BSD 3 clause

# %%
# Generating a dataset
# --------------------
#
# We create a synthetic dataset. The true generative process will take a 1-D
# vector and compute its sine. Note that the period of this sine is thus
# :math:`2 \pi`. We will reuse this information later in this example.
import numpy as np

from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.kernel_ridge import KernelRidge

rng = np.random.RandomState(0)
data = np.linspace(0, 30, num=1_000).reshape(-1, 1)
target = np.sin(data).ravel()

# %%
# Now, we can imagine a scenario where we get observations from this true
# process. However, we will add some challenges:
#
# - the measurements will be noisy;
# - only samples from the beginning of the signal will be available.
training_sample_indices = rng.choice(np.arange(0, 400), size=40, replace=False)
training_data = data[training_sample_indices]
training_noisy_target = target[training_sample_indices] + 0.5 * rng.randn(
    len(training_sample_indices)
)

# %%
# Let's plot the true signal and the noisy measurements available for training.
import matplotlib.pyplot as plt

plt.plot(data, target, label="True signal", linewidth=2)
plt.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
plt.legend()
plt.xlabel("data")
plt.ylabel("target")
_ = plt.title(
    "Illustration of the true generative process and \n"
    "noisy measurements available during training"
)

    
# %%
# We get a much more accurate model. We still observe some errors mainly due to
# the noise added to the dataset.
#
# Gaussian process regression
# ...........................
#
# Now, we will use a
# :class:`~sklearn.gaussian_process.GaussianProcessRegressor` to fit the same
# dataset. When training a Gaussian process, the hyperparameters of the kernel
# are optimized during the fitting process. There is no need for an external
# hyperparameter search. Here, we create a slightly more complex kernel than
# for the kernel ridge regressor: we add a
# :class:`~sklearn.gaussian_process.kernels.WhiteKernel` that is used to
# estimate the noise in the dataset.

import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel

kernel = 1.0 * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) + WhiteKernel(
    1e-1
)
gaussian_process = GaussianProcessRegressor(kernel=kernel)
start_time = time.time()
gaussian_process.fit(training_data, training_noisy_target)
print(
    f"Time for GaussianProcessRegressor fitting: {time.time() - start_time:.3f} seconds"
)

# %%
# The computation cost of training a Gaussian process is much less than the
# kernel ridge that uses a randomized search. We can check the parameters of
# the kernels that we computed.
gaussian_process.kernel_

# %%
# Indeed, we see that the parameters have been optimized. Looking at the
# `periodicity` parameter, we see that we found a period close to the
# theoretical value :math:`2 \pi`. We can have a look now at the predictions of
# our model.
start_time = time.time()
mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(
    data,
    return_std=True,
)
print(
    f"Time for GaussianProcessRegressor predict: {time.time() - start_time:.3f} seconds"
)


# %%
# We observe that the results of the kernel ridge and the Gaussian process
# regressor are close. However, the Gaussian process regressor also provide
# an uncertainty information that is not available with a kernel ridge.
# Due to the probabilistic formulation of the target functions, the
# Gaussian process can output the standard deviation (or the covariance)
# together with the mean predictions of the target functions.
#
# However, it comes at a cost: the time to compute the predictions is higher
# with a Gaussian process.
#
# Final conclusion
# ----------------
#
# We can give a final word regarding the possibility of the two models to
# extrapolate. Indeed, we only provided the beginning of the signal as a
# training set. Using a periodic kernel forces our model to repeat the pattern
# found on the training set. Using this kernel information together with the
# capacity of the both models to extrapolate, we observe that the models will
# continue to predict the sine pattern.
#
# Gaussian process allows to combine kernels together. Thus, we could associate
# the exponential sine squared kernel together with a radial basis function
# kernel.
from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * ExpSineSquared(1.0, 5.0, periodicity_bounds=(1e-2, 1e1)) * RBF(
    length_scale=15, length_scale_bounds="fixed"
) + WhiteKernel(1e-1)
gaussian_process = GaussianProcessRegressor(kernel=kernel)
gaussian_process.fit(training_data, training_noisy_target)
mean_predictions_gpr, std_predictions_gpr = gaussian_process.predict(
    data,
    return_std=True,
)

# %%
plt.plot(data, target, label="True signal", linewidth=2, linestyle="dashed")
plt.scatter(
    training_data,
    training_noisy_target,
    color="black",
    label="Noisy measurements",
)
# Plot the predictions of the gaussian process regressor
plt.plot(
    data,
    mean_predictions_gpr,
    label="Gaussian process regressor",
    linewidth=2,
    linestyle="dotted",
)
plt.fill_between(
    data.ravel(),
    mean_predictions_gpr - std_predictions_gpr,
    mean_predictions_gpr + std_predictions_gpr,
    color="tab:green",
    alpha=0.2,
)
plt.legend(loc="lower right")

plt.xlabel("data")
plt.ylabel("target")
_ = plt.title("Effect of using a radial basis function kernel")

# %%
# The effect of using a radial basis function kernel will attenuate the
# periodicity effect once that no sample are available in the training.
# As testing samples get further away from the training ones, predictions
# are converging towards their mean and their standard deviation
# also increases.