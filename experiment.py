import numpy as np
from functions import *
import sys
import multiprocessing as mp
import datetime
from sklearn import mixture
import logging

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=1, type=int, help='Model number.')
parser.add_argument('-K', '--K', default=3, type=int, help='Model number.')
parser.add_argument('-np','--nproc', default=12, type=int, help='Number of processes to run in parallel.')
parser.add_argument('-r' ,'--reps', default=20, type=int, help='Number of replications per sample size.')
parser.add_argument('-mi','--maxit', default=2000, type=int, help='Maximum EM iterations.')
parser.add_argument('-e', '--eps', default=1e-8, type=float, help='EM stopping criterion.')
args = parser.parse_args()

print(args)

model = args.model                    # Model number
n_proc = args.nproc                   # Number of cores to use
max_iter = args.maxit                 # Maximum EM iterations
eps = args.eps                        # EM Stopping criterion.
num_init = 5                          # Number of EM initializations
reps = args.reps                      # Number of replications to run per sample size
K = args.K

exec(open("models.py").read())

logging.basicConfig(filename='std_mod' + str(model) + '_K' + str(K) + '.log', filemode='w', format='%(asctime)s %(message)s')

print(ns)
print("Chose Model " + str(model))
print(model)

def sample(n):
    """ Sample from the mixture. """
    theta, sigma, pi = get_params(n)
    return sample_mixture(theta, sigma, pi, n)

def init_params(n, K):
    """ Starting values for EM algorithm. """
    theta0, sigma0, pi0 = get_params(n)

    theta_start = np.empty([K,d])
    sigma_start = np.empty([K,d,d])
    pi_start    = np.empty([K])

    inds = range(K0)

    # Make a partition of starting values near the true components.
    while True:
        s_inds = np.random.choice(inds, size=K)
        unique,counts = np.unique(s_inds, return_counts=True)

        if unique.size==K0:
            break
    
    for i in range(K):    
        if mix_type == "weak":
            theta_start[i,:]   = theta0[s_inds[i],:] + np.random.normal(0, 0.005*n**(-0.083), size=d).reshape((1,d))

        else:
            theta_start[i,:]   = theta0[s_inds[i],:] + np.random.normal(0, n**(-0.25), size=d).reshape((1,d))

        if mix_type == "weak":
            sigma_start[i,:,:] = sigma0[s_inds[i],:,:] + np.diag(np.abs(np.random.normal(0, 0.0005*n**(-0.25), size=d)))

        else:
            sigma_start[i,:,:] = sigma0[s_inds[i],:,:] 

        pi_start[i]        = pi0[s_inds[i]]/counts[s_inds[i]]

    return (theta_start, sigma_start, pi_start)
        
def process_chunk(bound):
    """ Run EM on a range of sample sizes. """
    ind_low = bound[0]
    ind_high= bound[1]

    m = ind_high - ind_low

    seed_ctr = 2000 * ind_low   # Random seed

    chunk_theta = np.empty((m, reps, K,d))
    chunk_sigma = np.empty((m, reps, K,d,d))
    chunk_pi    = np.empty((m, reps, K))
    chunk_iters = np.empty((m, reps))
    run_out = np.empty((num_init, 5))

    for i in range(ind_low, ind_high):
        n = int(ns[i])
        xi = get_xi(ns[i])

        for rep in range(reps):
            np.random.seed(seed_ctr)
            X = sample(n)

            np.random.seed(seed_ctr+1)
            theta_start, sigma_start, pi_start = init_params(n,K)

            out = em(X, theta_start, sigma_start, pi_start, max_iter=max_iter, eps=eps, mix_type=mix_type, xi=xi)

            logging.warning('Model ' + str(model) + ', rep:' + str(rep) + ', n:' + str(n) + ", nind:" + str(i) + ", iters:" + str(out[-1]))
        
            chunk_theta[i-ind_low, rep, :, :]    = out[0]
            chunk_pi[i-ind_low, rep, :]          = out[2]
            chunk_sigma[i-ind_low, rep, :, :, :] = out[1]   
            chunk_iters[i-ind_low, rep]          = out[3]   

            seed_ctr += 1

    return (chunk_theta, chunk_sigma, chunk_pi, chunk_iters)

proc_chunks = []

Del = n_num // n_proc 

for i in range(n_proc):
    if i == n_proc-1:
        proc_chunks.append(( (n_proc-1) * Del, n_num) )

    else:
        proc_chunks.append(( (i*Del, (i+1)*Del ) ))

if n_proc == 12:
    proc_chunks = [(0, 25), (25, 40), (40, 50), (50, 60), (60, 67), (67, 75), (75, 80), (80, 85), (85, 90), (90, 94), (94, 97), (97, 100)]

else:
    proc_chunks = [(0, 12), (12, 20), (20, 25), (25, 30), (30, 35), (35, 39), (39, 42), (42, 45)]

with mp.Pool(processes=n_proc) as pool:
    proc_results = [pool.apply_async(process_chunk,
                                     args=(chunk,))
                    for chunk in proc_chunks]

    result_chunks = [r.get() for r in proc_results]

done_theta = np.concatenate([result_chunks[j][0] for j in range(n_proc)], axis=0)
done_sigma = np.concatenate([result_chunks[j][1] for j in range(n_proc)], axis=0)
done_pi    = np.concatenate([result_chunks[j][2] for j in range(n_proc)], axis=0)
done_iters = np.concatenate([result_chunks[j][3] for j in range(n_proc)], axis=0)

np.save("results/result_model" + str(model) +"_K" + str(K) + "_theta.npy", done_theta)
np.save("results/result_model" + str(model) +"_K" + str(K) + "_pi.npy", done_pi)
np.save("results/result_model" + str(model) +"_K" + str(K) + "_iters.npy", done_iters)

if mix_type != "si":
    np.save("results/result_model" + str(model) +"_K" + str(K) + "_sigma.npy", done_sigma)
