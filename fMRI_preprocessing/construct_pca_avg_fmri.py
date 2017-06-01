import numpy as np
from numpy import diag
import pickle as p
from numpy.linalg import svd, norm
import os
import pylab as pl
from scipy import stats
from scipy.stats import pearsonr, sem
from scipy.io import loadmat
from random import shuffle
from path_to_RESULTS import PATH_TO_RESULTS

avg_fmri_path = PATH_TO_RESULTS + 'RESULTS/AVG_fMRI/'
mask_types = ["PPHG_3mm_thr20", "a1plus_3mm","erez_a1_network","erez_dmna_network","erez_dmnb_network","erez_dorslan_network","erez_ventlan_network","Occipital_Lobe_3mm_thr25","v1plus_3mm"]
avg_filename = 'avgfMRI.npz'

save_path = PATH_TO_RESULTS + 'RESULTS/PCA_fMRI/'
for mask in mask_types:
	path = avg_fmri_path + mask + '/' + avg_filename
	avg = np.load(path)
	avg = avg['avgfMRI'] # voxels x timesteps
	U, s, V = svd(avg, full_matrices=False)
	# 20 dims for comparsion to SRM
	S = diag(s[0:20])
	pca = S.dot(V[0:20, :])
	print pca.shape
	out = save_path + mask + '/'
	if not os.path.exists(out):
		os.makedirs(out)
	print "Saving to outpath: " + out
	np.savez(out + 'pcafMRI.npz', pcafMRI=pca)



