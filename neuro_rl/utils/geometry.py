import itertools
import numpy as np

def compute_tangling(X: np.array, dt: float):
    
    # compute derivatives of X
    X_dot = np.diff(X, axis=0) / dt
    X_dot = np.insert(X_dot, 0, (X[0,:] - X[-1,:] ) / dt, 0)

    # compute constant, prevents denominator from shrinking to zero
    epsilon = 0.1 * np.var(X)

    # get number of timesteps
    N_T = np.shape(X)[0]

    # get pairwise combinations of all timesteps
    # 2nd answer: https://stackoverflow.com/questions/464864/get-all-possible-2n-combinations-of-a-list-s-elements-of-any-length
    C = np.array(list(itertools.combinations(range(N_T), 2)))

    # initialize all arrays
    C_t = np.zeros((N_T,2))
    X_diff_t = np.zeros((N_T,1))
    X_dot_diff_t = np.zeros((N_T,1))
    Q = np.zeros((N_T,1), dtype=float)

    # iterate over each timestep, t
    for t in range(N_T):

        # get indices for all time-wise pair (specific t, all t')
        C_t = C[np.any(C==t, axis=1),:]

        # || x_dot(t) - x_dot(t') || ^2 for all (specific t, all t')
        X_dot_diff_t = np.sum ( np.square( X_dot[C_t[:,0],:] - X_dot[C_t[:,1],:] ) , axis=1)

        # || x(t) - x(t') || ^2 for all (specific t, all t')
        X_diff_t = np.sum ( np.square( X[C_t[:,0],:] - X[C_t[:,1],:] ) , axis=1)
        
        # compute Q(t)
        Q[t] = np.max ( X_dot_diff_t / ( X_diff_t + epsilon) )

    return Q