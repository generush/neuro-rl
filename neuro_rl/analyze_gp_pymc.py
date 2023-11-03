import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pymc3 as pm
import theano.tensor as tt

# %config InlineBackend.figure_format = 'retina'
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
az.style.use("arviz-darkgrid")

def weinland(t, c, tau=4):
    return (1 + tau * t / c) * np.clip(1 - t / c, 0, np.inf) ** tau

def angular_distance(x, y, c):
    # https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    return (x - y + c) % (c * 2) - c
    
C = np.pi
x = np.linspace(0, C)

plt.figure(figsize=(16, 9))
for tau in range(4, 10):
    plt.plot(x, weinland(x, C, tau), label=f"tau={tau}")
plt.legend()
plt.ylabel("K(x, y)")
plt.xlabel("dist")

plt.plot(
    np.linspace(0, 10 * np.pi, 1000),
    abs(angular_distance(np.linspace(0, 10 * np.pi, 1000), 1.5, C)),
)
plt.ylabel(r"$\operatorname{dist}_{geo}(1.5, x)$")
plt.xlabel("$x$")

angles = np.linspace(0, 2 * np.pi)
observed = dict(x=np.random.uniform(0, np.pi * 2, size=5), y=np.random.randn(5) + 4)


def plot_kernel_results(Kernel):
    """
    To check for many kernels we leave it as a parameter
    """
    with pm.Model() as model:
        cov = Kernel()
        gp = pm.gp.Marginal(pm.gp.mean.Constant(4), cov_func=cov)
        lik = gp.marginal_likelihood("x_obs", X=observed["x"][:, None], y=observed["y"], noise=0.2)
        mp = pm.find_MAP()
        # actual functions
        y_sampled = gp.conditional("y", angles[:, None])
        # GP predictions (mu, cov)
        y_pred = gp.predict(angles[:, None], point=mp)
        trace = pm.sample_posterior_predictive([mp], var_names=["y"], samples=100)
    plt.figure(figsize=(9, 9))
    paths = plt.polar(angles, trace["y"].T, color="b", alpha=0.05)
    plt.scatter(observed["x"], observed["y"], color="r", alpha=1, label="observations")
    plt.polar(angles, y_pred[0], color="black")
    plt.fill_between(
        angles,
        y_pred[0] - np.diag(y_pred[1]) ** 0.5,
        y_pred[0] + np.diag(y_pred[1]) ** 0.5,
        color="gray",
        alpha=0.5,
        label=r"$\mu\pm\sigma$",
    )
    plt.fill_between(
        angles,
        y_pred[0] - np.diag(y_pred[1]) ** 0.5 * 3,
        y_pred[0] + np.diag(y_pred[1]) ** 0.5 * 3,
        color="gray",
        alpha=0.25,
        label=r"$\mu\pm3\sigma$",
    )
    plt.legend()

def circular():
    tau = pm.Deterministic("τ", pm.Gamma("_τ", alpha=2, beta=1) + 4)
    cov = pm.gp.cov.Circular(1, period=2 * np.pi, tau=tau)
    return cov

plot_kernel_results(circular)

print ('hi')