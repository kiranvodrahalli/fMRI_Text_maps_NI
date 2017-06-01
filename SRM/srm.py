#!/usr/bin/env python

# Constrainted EM algorithm for Shared Response Model

# A Reduced-Dimension fMRI Shared Response Model
# Po-Hsuan Chen, Janice Chen, Yaara Yeshurun-Dishon, Uri Hasson, James Haxby, Peter Ramadge 
# Advances in Neural Information Processing Systems (NIPS), 2015. (to appear) 

# movie_data is a three dimensional matrix of size voxel x TR x nsubjs
# movie_data[:,:,m] is the data for subject m, which will be X_m^T in the standard 
# mathematic notation

# E-step:
# E_s   : nvoxel x nTR
# E_sst : nvoxel x nvoxel x nTR
# M-step:
# W_m   : nvoxel x nvoxel x nsubjs
# sigma_m2 : nsubjs 
# Sig_s : nvoxel x nvoxel 

import numpy as np, scipy, random, sys, math, os
from scipy import stats

def align(movie_data, working_path, nfeature, align_algo, randseed):
    print 'SRM',
    sys.stdout.flush()
  
    nsubjs = len(movie_data)
    for m in range(nsubjs):
        assert movie_data[0].shape[1] == movie_data[m].shape[1], 'numbers of TRs are different among subjects'
    nTR = movie_data[0].shape[1]
  
    current_file = working_path+align_algo+'_current.npz'

    nvoxel = np.zeros((nsubjs,),dtype=int)
    for m in xrange(nsubjs):
        nvoxel[m] = movie_data[m].shape[0] 
    bX = []
    trace_XtX = np.zeros((nsubjs))
    for m in range(nsubjs):
        bX.append(stats.zscore(movie_data[m].T ,axis=0, ddof=1).T)
        trace_XtX[m] = (bX[m]** 2).sum()

    del movie_data

    # initialization when first time run the algorithm
    if not os.path.exists(current_file):       
        bW = []
        bmu = []
        bSig_s = np.identity(nfeature)
        sigma2 = np.zeros(nsubjs)
        ES = np.zeros((nfeature, nTR))
  
        #initialization
        if randseed != None:
            print 'randinit',
            np.random.seed(randseed)
            for m in xrange(nsubjs):
                A = np.random.random((nvoxel[m],nfeature))
                Q, R_qr = np.linalg.qr(A)
                bW.append(Q)
        else:
            for m in xrange(nsubjs):
                Q = np.identity(nvoxel[m])
                bW.append(Q[:, :nfeature])
       
        for m in range(nsubjs):
            bmu.append(np.mean(bX[m], 1))
            sigma2[m] = 1

        niter = 0
        np.savez_compressed(working_path+align_algo+'_'+str(niter)+'.npz',\
                            bSig_s = bSig_s, bW = bW, bmu=bmu, sigma2=sigma2, ES=ES, niter=niter)
  
        # more iterations starts from previous results
    else:
        workspace = np.load(current_file)
        niter = workspace['niter']
        workspace = np.load(working_path+align_algo+'_'+str(niter)+'.npz')
        bSig_s = workspace['bSig_s'] 
        bW     = workspace['bW']
        bmu    = workspace['bmu']
        sigma2 = workspace['sigma2']
        ES     = workspace['ES']
        niter  = workspace['niter']

   
    for m in range(nsubjs):
        bX[m] = bX[m] - bX[m].mean(axis=1)[:,np.newaxis]

  
    print str(niter+1)+'th',
   
    (L_Sig_s, lwr) = scipy.linalg.cho_factor(bSig_s, check_finite=False)
    inv_Sig_s = scipy.linalg.cho_solve((L_Sig_s, lwr), np.identity(nfeature), check_finite=False)
    Sig_ss = inv_Sig_s + np.identity(nfeature) * ((1 / sigma2).sum())
    (L_Sig_ss, lower_ss) = scipy.linalg.cho_factor(Sig_ss, check_finite=False)
    invSig_ss = scipy.linalg.cho_solve((L_Sig_ss, lower_ss), np.identity(nfeature), check_finite=False)

    bWt_invsigma_X = np.zeros((nfeature, nTR))
    trace_bXt_invsigma2_X = 0.0

    for m in range(nsubjs):
        bWt_invsigma_X += (bW[m].T.dot(bX[m])) / sigma2[m]
        trace_bXt_invsigma2_X += trace_XtX[m] / sigma2[m]

    ES = bSig_s.dot(np.identity(nfeature) - np.sum(1 / sigma2) * invSig_ss).dot(bWt_invsigma_X)
    bSig_s = invSig_ss + ES.dot(ES.T) / float(nTR)
    det_psi = np.sum(np.log(sigma2) * nvoxel[0]) #TODO correct nvoxel here

    trSig_s = nTR * np.trace(bSig_s)
    for m in range(nsubjs):
        print ('.'),
        sys.stdout.flush()
        Am = bX[m].dot(ES.T)
        pert = np.zeros((Am.shape))
        np.fill_diagonal(pert, 1)
        Um, sm, Vm = np.linalg.svd(Am + 0.001 * pert, full_matrices=0)
        bW[m] = Um.dot(Vm)
        sigma2[m] = trace_XtX[m]
        sigma2[m] += -2 * np.sum(bW[m] * Am).sum()
        sigma2[m] += trSig_s
        sigma2[m] /= float(nTR * nvoxel[m])


    new_niter = niter + 1
    np.savez_compressed(current_file, niter = new_niter)  
    np.savez_compressed(working_path+align_algo+'_'+str(new_niter)+'.npz',\
                        bSig_s = bSig_s, bW = bW, bmu=bmu, sigma2=sigma2, ES=ES, niter=new_niter)
    os.remove(working_path+align_algo+'_'+str(new_niter-1)+'.npz')

    # calculate log likelihood
    logdet = np.log(np.diag(L_Sig_ss) ** 2).sum() + det_psi + np.log(np.diag(L_Sig_s) ** 2).sum()
    sign = -np.sign(logdet)
    if sign == -1:
        print ('log sign negative')

    loglike = -0.5 * nTR * logdet
    loglike += -0.5 * trace_bXt_invsigma2_X \
        + 0.5 * np.trace(bWt_invsigma_X.T.dot(invSig_ss).dot(bWt_invsigma_X))
             
    np.savez_compressed(working_path+align_algo+'_'+'loglikelihood_'+str(new_niter)+'.npz',\
                        loglike=loglike)
    
    print str(loglike) 
  
    return new_niter
