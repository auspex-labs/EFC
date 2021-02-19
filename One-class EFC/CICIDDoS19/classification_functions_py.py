import numpy as np
from dca_functions import *

def create_model(data,n_bins, lamb):
    sitefreq = Sitefreq(data, n_bins, lamb)
    pairfreq = Pairfreq(data, sitefreq, n_bins, lamb)
    couplingmatrix = Coupling(sitefreq, pairfreq, n_bins)
    h_i = local_fields(couplingmatrix, sitefreq, n_bins)

    return couplingmatrix, h_i

def test_model(test_data, model, h_i, expected, cutoff, n_bins):
    energies = []
    predicted = []
    for i in range(test_data.shape[0]):
        e = 0
        for j in range(test_data.shape[1]-1):
            for k in range(j,test_data.shape[1]):
                if test_data[i,j] != (n_bins-1) and test_data[i,k] != (n_bins-1):
                    e -= np.log(model[int(j*(n_bins-1) + test_data[i,j]), int(k*(n_bins-1) + test_data[i,k])])
            if test_data[i,j] != (n_bins-1):
                e -= np.log(h_i[int(j*(n_bins-1) + test_data[i,j])])
        if (e <= cutoff) and (expected[i] == 0):
            predicted.append(0)
        elif e <= cutoff:
            predicted.append(0)
            print(e, test_data[i,:])
        elif (e > cutoff) and (expected[i] != 0):
            predicted.append(1)
        elif (e > cutoff):
            predicted.append(1)
        energies.append(e)
    return predicted, energies


def define_cutoff(train_data, h_i, couplingmatrix, n_bins):
    energies = []
    for i in range(train_data.shape[0]):
        e = 0
        for j in range(train_data.shape[1]-1):
            for k in range(j,train_data.shape[1]):
                if train_data[i,j] != (n_bins-1) and train_data[i,k] != (n_bins-1):
                    e -= np.log(couplingmatrix[int(j*(n_bins-1) + train_data[i,j]), int(k*(n_bins-1) + train_data[i,k])])
            if train_data[i,j] != (n_bins-1):
                e -= np.log(h_i[int(j*(n_bins-1) + train_data[i,j])])
        energies.append(e)

    energies.sort()
    cutoff = energies[int(len(energies)*0.99)]
    return cutoff