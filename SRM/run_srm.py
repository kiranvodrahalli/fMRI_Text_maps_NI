#!/usr/bin/env python

# This is the code to run SRM only
# Please refer to --help for arguments setting
# There is also an option to input matlab files directly. The input and output files
# will be in the same format.
#
# align_algo must be spatial_srm, srm_noneprob_kernel, srm_nonprob, or srm
#
# by Cameron Po-Hsuan Chen and Hejia Zhang @ Princeton

from path_to_RESULTS import PATH_TO_RESULTS
import numpy as np, scipy, random, sys, math, os, copy
import scipy.io
from scipy import stats
#import argparse
#import importlib
#import pprint
#import form_WS_matrix
from srm import align

mask_types = ["PPHG_3mm_thr20", "a1plus_3mm","erez_a1_network","erez_dmna_network","erez_dmnb_network","erez_dorslan_network","erez_ventlan_network","Occipital_Lobe_3mm_thr25","sherlock1FR_17ss_ISC_thr0.2","v1plus_3mm"]


# WILL NOT CURRENTLY WORK ON DROPBOX (no original fmri data path)
#main_path = '/jukebox/norman/knv/thesis/sherlock/RESULTS/'
main_path = PATH_TO_RESULTS + 'RESULTS/'

def run_SRM_all(ndims, tr_times, tst_times):
    W_test = {}
    for mask in mask_types:
        output = run_SRM(ndims, mask, tr_times, tst_times)
        if output != None:
            W, S, test_data = output
            W_test[mask] = (W, S, test_data)
    return W_test

#ndims = 20, 50, 80, 100 etc.
def run_SRM(ndims, mask_type, tr_times, tst_times):
    dataset = str(ndims) + 'dim-SRM-' + mask_type
    align_algo = 'srm' # don't change - form_WS_matrix code required for other algos
    nfeature = ndims # 20dim
    niter = 10 # enough
    randseed = 0
    fmat = 'NPZ'
    strfresh = True


    # rondo options
    output_path = main_path + 'SRM_fMRI/' + mask_type + '/'
    input_path = '/tigress/knv/sherlock/fMRI_masked/'
    working_path = '/tigress/knv/fastscratch/srm' + str(ndims) + 'dim-sherlock-new/'



    # load data for alignment
    # let's modify this part instead to the construct_SRM_input_sherlock setup
    print 'start loading data'
    if fmat == 'NPZ':  # If the data is in .npz
        fmri_path = input_path
        fmri_data_type = 'sherlock_movie_s'
        npz = '.npz'
        assert(mask_type != '.DS_Store' and mask_type != 'README')
        print "Mask type: " + mask_type
        mask_data = []
        test_data = []
        for i in range(1, 18): # 17 total
            if i != 5:
                print "Subject # " + str(i)
                data = np.load(fmri_path + mask_type + '/' + fmri_data_type + str(i) + npz)
                data = data['data'] # num_voxels_for_given_mask x 1976 
                data = data[:, 3:] # REMOVE FIRST 3 TIME POINTS, we WILL NOT USE THEM!
                print "Shape: " + str(data.shape)
                tr_data = data[:, tr_times] #data[:, :1973/2]
                tst_data = data[:, tst_times] #data[:, 1973/2:]
                mask_data.append(tr_data)
                test_data.append(tst_data)
        data = mask_data
        # zscore and sanity check
        nsubjs = len(data)
        nvoxel = np.zeros((nsubjs,))
        nTR = data[0].shape[1]
        align_data = []
        if data[0].shape[0] < nfeature :
            print "skipping " + mask_type + " because dimensions are too small for selected dimension reduction"
            return None
        else:
            print "this mask has ok # of voxels" 
            print "nfeature = " + str(nfeature) + ", data[0].shape[0] = " + str(data[0].shape[0])
        for m in range(nsubjs):
            assert data[0].shape[1] == data[m].shape[1], 'numbers of TRs are different among subjects'
            nvoxel[m] = data[m].shape[0]
            assert nvoxel[m] >= nfeature, 'number of features is larger than number of voxels'
            print "Subject #" + str(m)
            print "Does this subject have zeros?: "
            print(np.where(~data[m].any(axis=1))[0])
            tmp = stats.zscore(data[m].T, axis=0, ddof=1).T
            nanidx = np.unique(np.where(np.isnan(tmp)==True)[0])
            if len(nanidx) != 0:
                 print  'You have voxels with all zeros. Please remove those voxels. \n  Subject {} Voxel {}'.format(m,nanidx)
            align_data.append(tmp)  

        # creating working folder
        if not os.path.exists(working_path):
            os.makedirs(working_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)    

        if strfresh:
            if os.path.exists(working_path + align_algo +'_current.npz'):
                os.remove(working_path + align_algo +'_current.npz')    
    

        # run alignment
        print 'start alignment'
        #algo = importlib.import_module('alignment_algo.'+align_algo)
        if os.path.exists(working_path+align_algo+'_current.npz'):
          workspace = np.load(working_path+align_algo+'_current.npz')
          new_niter = workspace['niter']
        else:
          new_niter = 0 

        while (new_niter<niter):  
            new_niter = align(align_data, working_path, nfeature, align_algo, randseed) 
    

        # form WS matrix
        print 'start transform'
        workspace = np.load(working_path+align_algo+'_'+str(niter)+'.npz')


        # using 'srm' for align_algo
        W = workspace['bW']
        S  = workspace['ES'] 

        if fmat == 'NPZ':
            # save the align_data too (this is like the original data after being zscored)!
            np.savez_compressed(output_path + 'WsS_' + str(ndims) + 'dim_tr_tst.npz',
                                Ws=W, S=S, tr=align_data, tst=test_data)
            

        print 'alignment done'
        # S is projected training data
        # need to project test data into shared space with Ws
        return W, S, test_data

if __name__ == '__main__':
    srm_dims = [20, 50, 80, 100]
    for dim in srm_dims:
        run_SRM_all(dim, xrange(1973/2+1), xrange(1973/2+1, 1973))
