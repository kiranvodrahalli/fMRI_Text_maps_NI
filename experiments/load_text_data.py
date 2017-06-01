import numpy as np
from path_to_RESULTS import PATH_TO_RESULTS

wiki_RW_vec_path = ''
skipthought_path = ''
conv_tensor_path = ''
atom_dict_path = ''

#annotation_vec_path = '/jukebox/norman/knv/thesis/sherlock/RESULTS/SEMVECS/'
annotation_vec_path = PATH_TO_RESULTS + 'RESULTS/SEMVECS/'

# original one, without any messing around (what I got original results with)
may15_no_auto = 'atoms_TRs_100dim_weighted_top4_dense_vec_pmweights.npz'

# inverse frequency weighted word vectors # Sept 12, yingyu + sanjeev suggestions
sep12_weighted_wvs = 'avg_100dim_wordvec_mat_Sep12_weighted.npz'

# no additional weighting, just word vector sum
sep12_unweighted_wvs = 'avg_100dim_wordvec_mat_Sep12_unweighted.npz'

# k-sparse dictionary representation of the word vector weighted averages
# append the sparsity before, this is just the tag
sep12_sparse_weighted_wvs = 'sparse_avg_100dim_wordvec_mat_Sep12_weighted.npz'

# 3-sparse dictionary representation of the word vector unweighted averages
sep12_sparse_unweighted_wvs = 'sparse_avg_100dim_wordvec_mat_Sep12_unweighted.npz'

def load_may15_annotation_vecs():
	vecs = np.load(annotation_vec_path + may15_no_auto)
	vecs = vecs['atom_TRs']
	vecs = vecs.T # shape is 100 x 1976
	return vecs[:, 0:vecs.shape[1]-3] # REMOVE LAST 3 WORD VECTORS, will not use (no paired fMRI)

def load_sep12_weighted_word_vecs_annotations():
	vecs = np.load(annotation_vec_path + sep12_weighted_wvs)
	vecs = vecs['vecs']
	vecs = vecs.T # shape is 100 x 1976
	return vecs[:, 0:vecs.shape[1]-3] # REMOVE LAST 3 WORD VECTORS, will not use (no paired fMRI)

def load_sep12_unweighted_word_vecs_annotations():
	vecs = np.load(annotation_vec_path + sep12_unweighted_wvs)
	vecs = vecs['vecs']
	vecs = vecs.T # shape is 100 x 1976
	return vecs[:, 0:vecs.shape[1]-3] # REMOVE LAST 3 WORD VECTORS, will not use (no paired fMRI)

def load_sep12_sparse_weighted_word_vecs_annotations(sparsity):
	print "Sparsity = " + str(sparsity)
	if sparsity == 0:
		sparsity = ''
	vecs = np.load(annotation_vec_path + str(sparsity) + sep12_sparse_weighted_wvs)
	vecs = vecs['arr_0']
	assert(vecs.shape[0] == 100 and vecs.shape[1] == 1976)
	return vecs[:, 0:vecs.shape[1]-3] # REMOVE LAST 3 WORD VECTORS, will not use (no paired fMRI)

def load_sep12_sparse_unweighted_word_vecs_annotations(sparsity):
	print "Sparsity = " + str(sparsity)
	if sparsity == 0:
		sparsity = ''
	vecs = np.load(annotation_vec_path + str(sparsity) + sep12_sparse_unweighted_wvs)
	vecs = vecs['arr_0']
	assert(vecs.shape[0] == 100 and vecs.shape[1] == 1976)
	return vecs[:, 0:vecs.shape[1]-3] # REMOVE LAST 3 WORD VECTORS, will not use (no paired fMRI)

# helper function to subtract the mean of the column vectors
def subtract_column_mean(vecs):
	# average the columns
	avg_col_vec = vecs.mean(axis=1)
	avg_col_vec = np.ravel(avg_col_vec)
	print "avg_col_vec.shape = " + str(avg_col_vec.shape)
	# subtract the average from the columns
	new_vecs = vecs - avg_col_vec[:, np.newaxis]
	print "after subtracting mean, vecs.shape = " + str(new_vecs.shape)
	return new_vecs, avg_col_vec


if __name__ == '__main__':
	load_sep12_weighted_word_vecs_annotations()


