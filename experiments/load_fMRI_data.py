import numpy as np
from path_to_RESULTS import PATH_TO_RESULTS
#import pickle as p
#from numpy.linalg import svd, norm
#import os
#import pylab as pl
#from scipy import stats
#from scipy.stats import pearsonr, sem
#from scipy.io import loadmat
#from random import shuffle

#main_path = '/jukebox/norman/knv/thesis/sherlock/RESULTS/'
main_path = PATH_TO_RESULTS + 'RESULTS/'


avg_fmri_path = main_path + 'AVG_fMRI/'

pca_avg_fmri_path = main_path + 'PCA_fMRI/'

srm_fmri_path = main_path + 'SRM_fMRI/'

srmica_fmri_path = main_path + 'SRMICA_fMRI/'

# all masks
# mask_types = ["sherlock1FR_17ss_ISC_thr0.2", "PPHG_3mm_thr20", "a1plus_3mm","erez_a1_network","erez_dmna_network","erez_dmnb_network","erez_dorslan_network","erez_ventlan_network","Occipital_Lobe_3mm_thr25","v1plus_3mm"]

mask_types = ["sherlock1FR_17ss_ISC_thr0.2", "erez_dmna_network", "erez_dmnb_network","erez_dorslan_network","erez_ventlan_network","Occipital_Lobe_3mm_thr25"]

original_fmri = '/jukebox/norman/knv/thesis/sherlock/fMRI_masked/'
#original_fmri = main_path + 'fMRI_masked/' # CURRENTLY DOESNT EXIST IN DROPBOX

# goal, simply average the TRs
# assume fMRI has #columns = #TRs
def generate_average_fMRI_over_fixed_interval_size(fMRI, interval_size):
	num_TRs = fMRI.shape[1] # columns = TRs
	averaged_fMRI = []
	for i in range(0, num_TRs, interval_size):
		avg_TR = fMRI[:, i]/(0. + interval_size)
		endpoint = min(num_TRs, i + interval_size)
		for j in range(i+1, endpoint):
			avg_TR += fMRI[:, j]/(0. + interval_size)
		averaged_fMRI.append(avg_TR)
	averaged_fMRI = np.matrix(averaged_fMRI)
	averaged_fMRI = averaged_fMRI.T
	return averaged_fMRI


## NOTE: If NO averaging over fancy intervals, set average_over_TRs_interval_size to 1


def generic_load(path, fname, dict_name, average_over_TRs_interval_size):
	avgs = {}
	for mask in mask_types:
		avg = np.load(path + mask + '/' + fname + '.npz')
		avg = avg[dict_name]
		if average_over_TRs_interval_size > 1:
			avg = generate_average_fMRI_over_fixed_interval_size(avg, average_over_TRs_interval_size)
		avgs[mask] = avg
	return avgs

def load_avg(average_over_TRs_interval_size):
	return generic_load(avg_fmri_path, 'avgfMRI', 'avgfMRI', average_over_TRs_interval_size)

# make PCA consistent with SRM dimension
# add dimension here
def load_avg_pca(ndims, average_over_TRs_interval_size):
	return generic_load(pca_avg_fmri_path, str(ndims) + 'dim_pcafMRI', 'pcafMRI', average_over_TRs_interval_size)

# loads an original subject
# should also be unnecessary, since srm returns test_data from original subject as well as learned maps
def load_orig_fmri(subject, average_over_TRs_interval_size):
	return generic_load(original_fmri, 'sherlock_movie_s' + str(subject), 'data', average_over_TRs_interval_size)


# modify to load SRM with different dimension
# redo SRM training on the fly; don't save to file
# should be unnecessary if re-training the SRM and doing cross-validation
def load_srm(ndims, average_over_TRs_interval_size):
	srms = {}
	for mask in mask_types:
		srm = np.load(srm_fmri_path + mask + '/' + 'WsS_' + str(ndims) + 'dim_tr_tst.npz') 
		Ws = srm['Ws']
		S = srm['S'] # already averaged over subjects
		tr = srm['tr']
		tst = srm['tst'] # 16 subjects x num_vox x num_TRs
		num_TRs = tst.shape[2]
		# take into account averaging
		if average_over_TRs_interval_size > 1:
			S = generate_average_fMRI_over_fixed_interval_size(S, average_over_TRs_interval_size)
			# ceil(original_num_TRs/interval_size) = new # of TRs
			avg_tsts = np.zeros((tst.shape[0], tst.shape[1], (num_TRs/average_over_TRs_interval_size)+1))
			# iterate over subjects
			for i in range(tst.shape[0]):
				curr_sub_tst = tst[i, :, :]
				curr_sub_tst = generate_average_fMRI_over_fixed_interval_size(curr_sub_tst, average_over_TRs_interval_size)
				avg_tsts[i, :, :] = curr_sub_tst # need to fix size of third dimension
			tst = avg_tsts
		# note that we don't have to do anything to Ws, which has no dependence on the # of TRs

		# ignore the following comment - better to give both so that you can average over subsets?
		# might be better to just report averaging over all of them though (higher number)
		# might be reasonable to only load S (training) and [np.dot(Ws, tst[i]) for i in xrange(len(tst))]
		srms[mask] = (Ws, S, tst) # when testing, apply Ws to tst to project in the right space
	return srms

# modify to load SRM with different dimension
# redo SRM training on the fly; don't save to file
# should be unnecessary if re-training the SRM and doing cross-validation
def load_srmica(ndims, average_over_TRs_interval_size):
	srms = {}
	for mask in mask_types:
		if ndims == 20:
			# this is the original, which is 20 dimensional
			srm = np.load(srmica_fmri_path + mask + '/' + 'WsS_tr_tst.npz')
		else:
			srm = np.load(srmica_fmri_path + mask + '/' + 'WsS_' + str(ndims) + 'dim_tr_tst.npz') 
		Ws = srm['Ws']
		S = srm['S'] # already averaged over subjects
		tr = srm['tr']
		tst = srm['tst'] # 16 subjects x num_vox x num_TRs
		num_TRs = tst.shape[2]
		# take into account averaging
		if average_over_TRs_interval_size > 1:
			S = generate_average_fMRI_over_fixed_interval_size(S, average_over_TRs_interval_size)
			# ceil(original_num_TRs/interval_size) = new # of TRs
			avg_tsts = np.zeros((tst.shape[0], tst.shape[1], (num_TRs/average_over_TRs_interval_size)+1))
			# iterate over subjects
			for i in range(tst.shape[0]):
				curr_sub_tst = tst[i, :, :]
				curr_sub_tst = generate_average_fMRI_over_fixed_interval_size(curr_sub_tst, average_over_TRs_interval_size)
				avg_tsts[i, :, :] = curr_sub_tst # need to fix size of third dimension
			tst = avg_tsts
		# note that we don't have to do anything to Ws, which has no dependence on the # of TRs

		# ignore the following comment - better to give both so that you can average over subsets?
		# might be better to just report averaging over all of them though (higher number)
		# might be reasonable to only load S (training) and [np.dot(Ws, tst[i]) for i in xrange(len(tst))]
		srms[mask] = (Ws, S, tst) # when testing, apply Ws to tst to project in the right space
	return srms

