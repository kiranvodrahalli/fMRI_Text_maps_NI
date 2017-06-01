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
from mica import align_voxels as mica_align

mask_types = ["PPHG_3mm_thr20", "a1plus_3mm","erez_a1_network","erez_dmna_network","erez_dmnb_network","erez_dorslan_network","erez_ventlan_network","Occipital_Lobe_3mm_thr25","v1plus_3mm"]
#mask_types = ["sherlock1FR_17ss_ISC_thr0.2"] # run on 26000 voxels which have R > 0.2 intersubject timecourse correlation (i.e. things that are reasonably related)


def run_SRMICA_all(ndims, tr_times, tst_times):
    W_test = {}
    for mask in mask_types:
        W, S, test_data = run_SRMICA(ndims, mask, tr_times, tst_times)
        W_test[mask] = (W, S, test_data)
    return W_test

# ndims = number of dimensions
def run_SRMICA(ndims, mask_type, tr_times, tst_times):
    dataset = str(ndims) + 'dim-SRMICA-' + mask_type
    nfeature = ndims # 20dim
    niter = 10 # enough
    randseed = 0
    fmat = 'NPZ'
    strfresh = True


    # rondo options
    output_path = '/tigress/knv/sherlock/RESULTS/SRMICA_fMRI/' + mask_type + '/'
    input_path = '/tigress/knv/sherlock/fMRI_masked/'

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
                print 'You have voxels with all zeros. Please remove those voxels. \n  Subject {} Voxel {}'.format(m,nanidx)
            align_data.append(tmp)    
        # run alignment
        print 'start alignment'

        outputs = mica_align(align_data, niter, nfeature, randseed) 
        # outputs are [], [], bW, [], [], ES
        W = outputs[2]
        S = outputs[5]

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
        run_SRMICA_all(dim, xrange(1973/2+1), xrange(1973+1, 1973))
	

