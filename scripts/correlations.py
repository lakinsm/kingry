#!/usr/bin/env python3

import sys
import os
import numpy as np
import re
from scipy.stats.stats import pearsonr


mydir = os.getcwd()
kingry_file = mydir + '/../data/pre-filtering_kingry.csv'
kingry_names = mydir + '/../data/kingry_sample_names.txt'


def readKingryData():
    with open(kingry_file, 'r') as k, open(kingry_names, 'r') as n:
        X = np.array([x.split(',') for x in k.read().split('\n') if x], dtype=np.float32)
        names = np.array([x for x in n.read().split('\n') if x])
    return X, names


def selectGenesWithVariance(X):
    selected_rows = np.array([True if x > 500 else False for x in np.ptp(X, axis=1)], dtype=bool)
    return X[selected_rows, :], selected_rows


def worker(worker_data):

    def correlationsByGroup(X, outfile, name):
        with open(outfile, 'w') as out:
            for m in range(1, X.shape[0]):
                if m % 100 == 0:
                    sys.stdout.write('{}: {}\n'.format(name, m))
                if m != 1:
                    out.write('\n')
                # Across all timepoints within a tissue
                for n in range(m):
                    if n:
                        out.write(',')
                    out.write(str(pearsonr(X[m, :], X[n, :])[0]))
    X, outfile, name = worker_data
    correlationsByGroup(X, outfile, name)


if __name__ == '__main__':
    data, samples = readKingryData()
    data, idxs = selectGenesWithVariance(data)
    print(data.shape)
    np.savetxt(mydir + '/../data/16June2016_high_var_analytic_matrix_filtered.csv', data, delimiter=',')
    with open(mydir + '/../data/16June2016_high_var_indices_zero_based.csv', 'w') as idout:
        for e, i in enumerate(idxs):
            if i:
                idout.write('{},'.format(e))
    lvslung_idx = np.array([True if x in tuple(list(range(6)) + list(range(30, 54))) else False for x in range(data.shape[1])], dtype=bool)
    schulung_idx = np.array([True if x in tuple(list(range(30))) else False for x in range(data.shape[1])], dtype=bool)
    lvsspleen_idx = np.array([True if x in tuple(list(range(54, 60)) + list(range(84, 108))) else False for x in range(data.shape[1])], dtype=bool)
    schuspleen_idx = np.array([True if x in tuple(list(range(54, 84))) else False for x in range(data.shape[1])], dtype=bool)

    # LVS Lung
    parallel_inputs = ((data[:, lvslung_idx], mydir + '/../analyses/lvs_lung_corr_all_timepoints.txt', 'lvslung'),
                       (data[:, lvsspleen_idx], mydir + '/../analyses/lvs_spleen_corr_all_timepoints.txt', 'lvsspleen'),
                       (data[:, schulung_idx], mydir + '/../analyses/schu_lung_corr_all_timepoints.txt', 'schulung'),
                       (data[:, schuspleen_idx], mydir + '/../analyses/schu_spleen_corr_all_timepoints.txt', 'schuspleen'))

    mp.freeze_support()
    pool = mp.Pool(processes=4)
    pool.map(worker, parallel_inputs)
    pool.close()  # ask nicely
    pool.join()  # sigterm
    pool.terminate()  # sigkill
    del pool  # make sure pool is cleared
    print('Done.')


