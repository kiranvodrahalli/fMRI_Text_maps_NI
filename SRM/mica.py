#!/usr/bin/env python

# Multi-subject ICA for fMRI data alignment
# number of voxels should be the same for all subjects

import numpy as np
from scipy import stats
import warnings
from fastica import FastICA

def align_voxels(data, niter, nfeature, initseed):
    # "data" is now a list of subjects data
    nTR = data[0].shape[1]
    nsubjs = len(data)
    nvoxels = np.zeros((nsubjs,))
    for m in range(nsubjs):
        data[m] = np.nan_to_num(data[m])
    # zscore the data
    bX = np.empty(shape=(0,nTR))
    for m in range(nsubjs):
        nvoxels[m] = data[m].shape[0]
        bX = np.concatenate((bX,data[m]),axis=0)
    del data
    # record number of voxels for each subject
    vx_sum = np.zeros((nsubjs+1,))
    vx_sum[1:] = np.cumsum(nvoxels)
    bW = []
    # perform ICA
    np.random.seed(initseed)
    A = np.random.rand(nfeature,nfeature*nsubjs)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ica = FastICA(n_components= nfeature, max_iter=500,w_init=A,random_state=initseed,n_blocks=nsubjs, block_size=nvoxels[0])
            St = ica.fit_transform(bX.T)
            ES = St.T
            W = ica.mixing_
            # convert W to list
            for m in range(nsubjs):
                bW.append(W[vx_sum[m]:vx_sum[m+1],:])
        except (ValueError, np.linalg.linalg.LinAlgError):
            # print 'in except'
            for m in range(nsubjs):
                bW.append(np.eye(nvoxels[m],nfeature))
            ES = np.zeros((nfeature,nTR))

    return [], [], bW, [], [], ES