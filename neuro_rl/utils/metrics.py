import numpy as np

def tangling(X: np.array, t: np.array):
    
    dt = t[1] - t[0]
    
    # compute derivatives of X
    X_dot = np.diff(X, axis=0) / dt
    X_dot = np.insert(X_dot, 0, X_dot[0,:], axis=0)

    # find first time step
    first_indices = np.where(t == 0)[0]
    last_indices = np.roll(first_indices, -1) - 1
    last_indices[-1] = len(t) - 1

    for i in range(len(first_indices)):
        X_dot[first_indices[i], :] = ( X[first_indices[i], :] - X[last_indices[i], :] ) / dt
 
    # compute constant, prevents denominator from shrinking to zero
    epsilon = 0.1 * np.var(X)

    # Calculate the pairwise squared differences for X and X_dot
    # https://towardsdatascience.com/how-to-vectorize-pairwise-dis-similarity-metrics-5d522715fb4e
    X_diff_t = np.sum((X[:, None] - X[None, :]) ** 2, axis=-1)
    X_dot_diff_t = np.sum((X_dot[:, None] - X_dot[None, :]) ** 2, axis=-1)
    
    # Calculate the ratios of X_dot_diff to X_diff
    ratios = X_dot_diff_t / ( X_diff_t + epsilon )
    
    # Find the maximum ratio
    Q = ratios.max(axis=0)

    return Q
