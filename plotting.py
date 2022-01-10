import numpy as np
from functions import *
import argparse
import matplotlib.pyplot as plt
import matplotlib

n_num = 100
ns = np.concatenate([np.linspace(100, 900, 5), np.linspace(1000, 100000, n_num-5)])
n_num=ns.size

text_size = 17

matplotlib.rc('xtick', labelsize=text_size) 
matplotlib.rc('ytick', labelsize=text_size) 
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['text.latex.preamble']=[r"\usepackage{amsmath}"]

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

lw = 3
elw=0.8

def plot_model(K, model, n0=0, q=2):
    
    D = np.load("results/result_model" + str(model) +"_K" + str(K) + "_loss.npy")

    fig = plt.figure()
    
    loss        = np.mean(D, axis=1)
    yerr      = 2*np.std(D, axis=1)
    lab="temp"

    Y = np.array(np.log(loss)).reshape([-1,1])

    if model == 1:
        label = "$\mathcal{D}(\widehat G_n, G_0)$"

    elif model == 2:
        label = "$\\bar{\mathcal{D}}(\widehat G_n, G_0)$"

    else:
        label = "$\widetilde W(\widehat G_n, G_0^n)$"

    plt.errorbar(np.log(ns), Y, yerr=yerr, lw=lw, elinewidth=elw, label=label)
    plt.grid(True, alpha=.5)

    X = np.empty([ns.size-n0, 2])
    X[:,0] = np.repeat(1, ns.size-n0)
    X[:,1] = np.log(ns[n0:])
    Y = Y[n0:] 
        
    beta = (np.linalg.inv(X.transpose() @ X) @ X.transpose() @ Y)
    print("Beta: ", beta[1,0])

    plt.plot(X[:,1], X @ beta, lw=lw, label=str(np.round(beta[0,0], 1)) + "$n^{" + str(np.round(beta[1,0],2)) + "}$" )
    
    plt.xlabel("$\log n$", fontsize=text_size)
    plt.ylabel("Log Loss", fontsize=text_size)#"$\log$ " + lab)
    plt.legend(loc="upper right", title="", prop={'size': text_size})

    plt.savefig("plots/plot_model" + str(model) +"_K" + str(K) + "_n0_" + str(n0) + ".pdf", bbox_inches = 'tight',pad_inches = 0)

print("SI")
plot_model(model=1, K=3, n0=0, q=2) 
plot_model(model=1, K=4, n0=0, q=2) 

print("Gaussian")
plot_model(model=2, K=4, n0=0, q=4) 
plot_model(model=2, K=5, n0=0, q=6) 

print("Uniform SI")
plot_model(model=3, K=3, n0=0, q=3) 
plot_model(model=4, K=4, n0=0, q=5) 
