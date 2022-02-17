import numpy as np
import ot

def si_dist(theta0, theta):
    return np.linalg.norm(theta-theta0)

def gauss_dist(theta0, sigma0, theta, sigma):
    return np.linalg.norm(theta0 - theta) + np.linalg.norm(sigma0-sigma)

def si_loss(theta0, pi0, theta, pi):
    K0 = theta0.shape[0]
    K  = theta.shape[0]

    D = np.empty((K,K0))

    for i in range(K):
        for j in range(K0):
            D[i,j] = si_dist(theta[i,:], theta0[j,:])

    vor=[]
    for i in range(K):
        vor.append(np.argmin(D[i,:]))

    unique, counts = np.unique(vor, return_counts=True)
    d = dict(zip(unique, counts))

    summ = 0
    for i in range(K):
        if counts[vor[i]] == 1:
            summ += pi[i] * D[i,vor[i]]

        else:
            summ += pi[i] * D[i,vor[i]]**2

    for j in range(K0):
        jj = np.repeat(j, K)

        inds = np.argwhere(vor == jj)
        inds = inds.flatten()

        pi_bar = 0
        for ind in inds:
            pi_bar += pi[ind] 

        summ += np.abs(pi_bar - pi0[j])

    return summ

rbar = [0,1,4,6] # cf. Proposition 2.1 of [Ho and Nguyen (2016), Annals of Statistics].
def gauss_loss(theta0, sigma0, pi0, theta, sigma, pi):
    K0 = theta0.shape[0]
    K  = theta.shape[0]
    
    D = np.empty((K,K0))

    for i in range(K):
        for j in range(K0):
            D[i,j] = gauss_dist(theta0[j,:], sigma0[j,:,:], theta[i,:], sigma[i,:,:])

    vor=[]
    for i in range(K):
        for k in range(K0):
            if D[i,k] == np.min(D[i,:]):
                vor.append(k)

    unique, counts = np.unique(vor, return_counts=True)
    d = dict(zip(unique, counts))

    mask = ~np.eye(theta.shape[1],dtype=bool)
    summ = 0.0
    for i in range(K):
        j = vor[i]
        if counts[vor[i]] == 1:
            summ += pi[i] * D[i,vor[i]] 

        else:
            j = vor[i]
            rb = rbar[counts[j]]
            theta_dist = (np.linalg.norm(theta[i,:]-theta0[j,:]))**rb
            sigma_dist = (np.linalg.norm(sigma[i,:,:] - sigma0[j,:,:]))**(rb/2.0)
            summ += pi[i] * (theta_dist + sigma_dist)

    for k in range(K0):
        pi_bar = 0

        for i in range(K):
            if vor[i] == k:
                pi_bar += pi[i]

        summ += np.abs(pi_bar - pi0[k])

    return summ

def si_uniform_loss(theta_star, pi_star, theta0, pi0, theta, pi):
    K_star = theta_star.shape[0]
    K0     = theta0.shape[0]
    K      = theta.shape[0]

    D_t_star = np.empty((K,K_star))
    D_0_star = np.empty((K0,K_star))

    for i in range(K):
        for j in range(K_star):
            D_t_star[i,j] = si_dist(theta[i,:], theta_star[j,:])

    for i in range(K0):
        for j in range(K_star):
            D_0_star[i,j] = si_dist(theta0[i,:], theta_star[j,:])

    vor_t=[]
    vor_0=[]
    for i in range(K):
        vor_t.append(np.argmin(D_t_star[i,:]))

    for i in range(K0):
        vor_0.append(np.argmin(D_0_star[i,:]))

    unique_t, counts_t = np.unique(vor_t, return_counts=True)
    unique_0, counts_0 = np.unique(vor_0, return_counts=True)

    ot_dist = np.empty((K, K0))

    for i in range(K):
        for j in range(K0):
            if vor_t[i] == vor_0[j]:
                ot_dist[i,j] = (si_dist(theta[i,:], theta0[j,:]))**(counts_t[vor_t[i]] + counts_0[vor_0[j]] - 1)

            else:
                ot_dist[i,j] = 1

    return ot.emd2(pi, pi0, ot_dist)
