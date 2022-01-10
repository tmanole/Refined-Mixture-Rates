import numpy as np
from copy import deepcopy
from scipy.stats import multivariate_normal

def sample_mixture(theta0, sigma0, pi0, n):
    d = theta0.shape[1]
    x = np.empty((n,d))

    for i in range(n):
        u = np.random.uniform(size=1)

        this_pi = 0
        for j in range(pi0.size):
            this_pi += pi0[j]

            if u < this_pi:
                x[i,:] = np.random.multivariate_normal(mean=theta0[j,:], cov=sigma0[j,:,:], size=1)
                break 

    return x

def density(X, theta, sigma):
    out = np.empty((X.shape[0], theta.shape[0]))

    for i in range(theta.shape[0]):
        out[:,i] = multivariate_normal.pdf(X, mean=theta[i,:], cov=sigma[i,:,:])

    return out

def em(X, theta_start, sigma_start, pi_start, max_iter=5000, eps=1e-8, mix_type="weak", xi=3):

    theta_prev = theta_start
    sigma_prev = sigma_start
    pi_prev    = pi_start

    theta_new = theta_start
    sigma_new = sigma_start
    pi_new    = pi_start

    K = np.size(pi_start)
    n = X.shape[0]
    d = X.shape[1]

    T1 = [None] * K
    T2 = [None] * K
    T3 = [None] * K    

    for iiter in range(max_iter):

        density_eval = np.array(density(X, theta_new, sigma_new))
        weights = np.tile(pi_new, (n, 1)) * density_eval#.reshape([-1,1])
        weights /= np.tile(np.sum(weights, 1), (K, 1)).T

        for i in range(K):
            T1[i] = np.sum(weights[:,i])
            T2[i] = np.sum(weights[:,i].reshape([-1,1]) * X, axis=0).reshape([1,d])

        pi_new = [(T1[i] + xi) / (n + K * xi) for i in range(K)]

        for i in range(K):
            theta_new[i,:] = T2[i]/T1[i]

        if mix_type == "weak":
            # Use speedup for variance proposed by McLachlan and Peel (2000). 
            for i in range(K):
                T3 = np.dot(X.T, weights[:,i].reshape([-1,1]) * X)
                sigma_new[i,:,:] = (T3 - ((1.0/T1[i] ) * np.outer(T2[i], T2[i]))) / T1[i]

        if iiter > 500 or (iiter > 100 and iiter % 10 == 0):
           if np.linalg.norm(theta_new - theta_prev) + np.linalg.norm(sigma_new - sigma_prev) < eps or iiter > max_iter:
               break

        theta_prev = deepcopy(theta_new)
        sigma_prev = deepcopy(sigma_new)
        pi_prev    = deepcopy(pi_new)

    return (theta_new, sigma_new, pi_new, iiter)
