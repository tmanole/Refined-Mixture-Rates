import ot
import sys
import numpy as np
from discrepancies import *
from scipy.spatial.distance import cdist

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', default=1, type=int, help='Model number.')
parser.add_argument('-K', '--K', default=3, type=int, help='Model number.')
args = parser.parse_args()

print(args)

model = args.model               
K = args.K
reps = 20 

exec(open("models.py").read())

dists = np.empty((n_num, reps))

thetas = np.load("results/result_model" + str(model) +"_K" + str(K) + "_theta.npy")
pis    = np.load("results/result_model" + str(model) +"_K" + str(K) + "_pi.npy")

if mix_type != "si":
    sigmas = np.load("results/result_model" + str(model) +"_K" + str(K) + "_sigma.npy")

if model==1:
    mix_type = "si"
    q = 2

elif model == 2:
    mix_type = "weak"
    q = 4 if K==4 else 6

elif model == 3:
    mix_type = "uniform_si"
    q = 3

elif model == 4:
    mix_type = "uniform_si"
    q = 5

else:
    sys.exit("Model unrecognized.")
    
for i in range(n_num):
    theta0, sigma0, pi0 = get_params(ns[i])

    for j in range(reps):
        if mix_type == "si":
            dists[i,j] = si_loss(theta0, pi0, thetas[i,j,:,:], pis[i,j,:])

        elif mix_type == "uniform_si":
            dists[i,j] = si_uniform_loss(theta_star, pi_star, theta0, pi0, thetas[i,j,:,:], pis[i,j,:])

        else:
            dists[i,j] = gauss_loss(theta0, sigma0, pi0, thetas[i,j,:,:], sigmas[i,j,:,:,:], pis[i,j,:]) 

np.save("results/result_model" + str(model) +"_K" + str(K) + "_loss.npy", dists)
