import numpy as np

from path_to_RESULTS import PATH_TO_RESULTS
from linear_maps import procrustes_fit, learn_matrix_factorization_for_time_text_weights, create_k_diag_conv_mat_from_weights
from load_fMRI_data import load_avg, load_avg_pca, load_srm, load_srmica
from load_text_data import subtract_column_mean, load_may15_annotation_vecs, load_sep12_weighted_word_vecs_annotations, load_sep12_unweighted_word_vecs_annotations, load_sep12_sparse_weighted_word_vecs_annotations, load_sep12_sparse_unweighted_word_vecs_annotations
#from load_hmm_changepoints import changepoint_dict
from create_semantic_vectors import make_weighted_average_word_vecs_over_multiple_TRs
from performance_metrics import scene_classification, scene_ranking, voting_scene_classification, hmm_scene_classification, hmm_scene_ranking
from math import exp
from random import shuffle
from scipy.stats import sem

import pickle as p

# list of weightings to check
def gen_exp_weightings():
	weights_list = []
	for w in np.linspace(0.1, 10, num=100):
		xaxis = np.array(range(0, 10))
		weights = []
		for i in range(10):
			x = xaxis[i]
			weights.append(exp(-1*w*x))
		weights = np.array(weights)
		weights /= 0.0 + sum(weights)
		weights_list.append((w, weights))
	return weights_list

# TODO: modify to have weights = exp(-i*omega), normalized (then just optimize over omega)
# instead of (c1, c2, c3, etc)
# avg_variant = 'srm', 'srmica', 'pca'
# subtract_temporal_average_bool = pre-process the word vectors by subtracting out the temporal average
# if True
# learn_weights = True if you do Alternating Minimization to actually learn the embeddings
# learn_weights = False if you instead generate exp_weightings and test on all of them, and check the best
# timestep_weighting must be None if learn_weights = True
# otherwise, timestep_weighting is the weighting you want to use.
# fmri_or_text_time_weights is a string that if is "fmri", we learn convolution map C on the fMRI side things to combine (or not learn, test, depending on learn_weight's value)
# if it is "text", we learn convolution map C as a weighted combination of the word embeddings (or not learn, test, depending on the value of learn_weights)
# USE THIS ONE: allows for avg, pca, srm too
def semantic_time_weighting_experiment_50_50_split(fmri_or_text_time_weights, timestep_weighting, learn_weights, avg_variant, date_of_wordvecs, subtract_temporal_average_bool):
	# if we're learning the weights, then timestep_weighting is irrelevant
	if learn_weights == True:
		assert(timestep_weighting == None)
	assert((fmri_or_text_time_weights == "fmri") or (fmri_or_text_time_weights == "text"))
	if avg_variant == 'avg':
		fmris = load_avg(1) # 1 because there's no averaging (see load_fmri)
	elif avg_variant == 'pca':
		fmris = load_avg_pca(1)
	elif avg_variant == 'srm' or avg_variant == 'srmica':
		# here, srm means we don't do the testing over T iterations
		# we just use the average of all of them without error bars
		if avg_variant == 'srm':
			srms = load_srm(1)
		elif avg_variant == 'srmica':
			srms = load_srmica(1)
		# calculate the average for each mask, and replace with that
		fmris = {}
		for mask in srms.keys():
			#if mask != 'erez_dmna_network':
			#	continue
			srm = srms[mask] 
			Ws = srm[0] # the map
			S = srm[1] # averaged SRM-projected training data
			num_training_time_steps = S.shape[1]
			tst_data_list = srm[2] # test portion of the data, pre-selected
			#print "tst_data_list length = " + str(len(tst_data_list))
			num_subjs = len(tst_data_list)
			avg_srm_tst_data = None
			for i in xrange(num_subjs):
				#print "Subject #"+ str(i)
				tst_data = tst_data_list[i]
				#print "Test data shape = " + str(tst_data.shape)
				transformed = np.dot(Ws[i, :, :].T, tst_data)
				if i == 0:
					avg_srm_tst_data = transformed/(0. + num_subjs)
				else:
					avg_srm_tst_data += transformed/(0. + num_subjs)
			fmri_mask = np.c_[S, avg_srm_tst_data]
			fmris[mask] = fmri_mask

	else:
		print 'avg_variant is wrong'
		return
	if date_of_wordvecs == 'may15':
		semantic_vecs = load_may15_annotation_vecs()
	elif date_of_wordvecs == 'sep12weighted':
		semantic_vecs = load_sep12_weighted_word_vecs_annotations()
	elif date_of_wordvecs == 'sep12unweighted':
		semantic_vecs = load_sep12_unweighted_word_vecs_annotations()
	else:
		# default
		semantic_vecs = load_sep12_weighted_word_vecs_annotations()
	print "Semantic vector shape = " + str(semantic_vecs.shape)
	num_sem_vecs = semantic_vecs.shape[1] # assume time is # cols
	# don't subtract out average of all semantic vectors for all time points
	# only calculate average for training, and subtract that average out of the test
	# CHANGE MADE: sep 19 12:40 AM
	'''
	# subtract out average if we're doing this case
	if subtract_temporal_average_bool == True:
		semantic_vecs, avg_semantic_vec = subtract_column_mean(semantic_vecs)
	'''

	#print "num semantic vectors = " + str(num_sem_vecs)
	word_tr = semantic_vecs[:, xrange(num_sem_vecs/2+1)]
	#print "word_tr.shape = " + str(word_tr.shape)
	word_tst = semantic_vecs[:, xrange(num_sem_vecs/2+1, num_sem_vecs)]
	#print "word_tst.shape = " + str(word_tst.shape)
	#print "word_tr.shape = " + str(word_tr.shape)
	#print "word_tst.shape = " + str(word_tst.shape)

	# don't subtract out average of all semantic vectors for all time points
	# only calculate average for training, and subtract that average out of the test
	# CHANGE MADE: sep 19 12:40 AM
	if subtract_temporal_average_bool == True:
		word_tr, avg_tr_word_vec = subtract_column_mean(word_tr)
		word_tst = word_tst - avg_tr_word_vec[:, np.newaxis]

	mask_results = {}
	for mask in fmris.keys():
		fmri = fmris[mask] 

		num_total_time_steps = fmri.shape[1]
		#print "fmri.shape = " + str(fmri.shape)
		fmri_tr = fmri[:, xrange(num_sem_vecs/2+1)]
		fmri_tst = fmri[:, xrange(num_sem_vecs/2 + 1, num_sem_vecs)]
		#print "fmri_tr.shape = " + str(fmri_tr.shape)
		#print "fmri_tst.shape = " + str(fmri_tst.shape)
		print "Training linear maps..."
		# all procrustes
		if learn_weights == True:
			if fmri_or_text_time_weights == "fmri":
				W_tf, W_ft, C_weights = learn_matrix_factorization_for_time_text_weights(xrange(num_sem_vecs/2+1), [], fmri_tr, mask + avg_variant, word_tr, date_of_wordvecs)
			elif fmri_or_text_time_weights == "text":
				W_tf, W_ft, C_weights = learn_matrix_factorization_for_time_text_weights(xrange(num_sem_vecs/2+1), [], word_tr, date_of_wordvecs, fmri_tr, mask + avg_variant)


		else:
			C_weights = timestep_weighting
			num_times = len(xrange(num_sem_vecs/2 + 1))
			C = create_k_diag_conv_mat_from_weights(np.zeros((num_times, num_times)), C_weights, num_times)
			

			if fmri_or_text_time_weights == "fmri":
				X = fmri_tr[:, xrange(num_sem_vecs/2 + 1)]
				Y = word_tr[:, xrange(num_sem_vecs/2 + 1)]
				YC = np.dot(Y, C)
				# text -> fmri
				W_tf = procrustes_fit([X, YC])
				W_tf = W_tf[0]
				W_ft = W_tf.T
			elif fmri_or_text_time_weights == "text":
				Y = fmri_tr[:, xrange(num_sem_vecs/2 + 1)]
				X = word_tr[:, xrange(num_sem_vecs/2 + 1)]
				YC = np.dot(Y, C)
				# text -> fmri
				W_tf = procrustes_fit([YC, X])
				W_tf = W_tf[0]
				W_ft = W_tf.T



		print "Done training linear maps..."

		
		
		TF_class_scores = []
		TF_rank_scores = []
		FT_class_scores = []
		FT_rank_scores = []

		num_time_steps = fmri_tst.shape[1] # assume time steps are # of columns
		chunk_array = []
		index = 0
		num_chunks = 25 ### CHANGED JAN 11 
		while index < num_time_steps:
			chunk_array.append(index)
			index += num_time_steps/num_chunks
			if index > num_time_steps:
				index = num_time_steps
		k = 1 #-> 4% chance rate ## CHANGED FEB 4

		# weighting matrices for semantic vectors
		C = create_k_diag_conv_mat_from_weights(np.zeros((num_time_steps, num_time_steps)), C_weights, num_time_steps)
		time_weighted_word_tst = np.dot(word_tst, C)


		# comparisons in fMRI space (i.e. text -> fMRI) 
		# truth is fmri_tst
		# prediction is W_tf*time_weighted_word_tst
		TF_truth = fmri_tst

		TF_prediction = np.dot(W_tf, time_weighted_word_tst)
		#TF_prediction = TF_prediction[0, :, :]
		#print "TF pred shape: " + str(TF_prediction.shape)
		#print "TF truth shape: " + str(TF_truth.shape)
		assert(TF_truth.shape == TF_prediction.shape)
			
		TF_classification_score = scene_classification(TF_truth, TF_prediction, chunk_array, k)
		TF_rank_score, TF_ranks = scene_ranking(TF_truth, TF_prediction, chunk_array)
		TF_rank_score = TF_rank_score/(num_chunks + 0.)

		
		# comparisons in Semantic space (i.e. fMRI -> text) (time-weighted_averages)
		# truth is time_weighted_word_tst
		# prediction is W_ft*fmri_tst
		FT_truth = time_weighted_word_tst
		FT_prediction = np.dot(W_ft, fmri_tst)
		#FT_prediction = FT_prediction[0, :, :]
		#print "FT pred shape: " + str(FT_prediction.shape)
		#print "FT truth shape: " + str(FT_truth.shape)
		assert(FT_truth.shape == FT_prediction.shape)

		FT_classification_score = scene_classification(FT_truth, FT_prediction, chunk_array, k)
		FT_rank_score, FT_ranks = scene_ranking(FT_truth, FT_prediction, chunk_array)
		FT_rank_score = FT_rank_score/(num_chunks + 0.)

		
		print "Mask: " + mask
		print "---------------------------------------------------------------------------"
		print "fMRI -> Text (Procrustes) scene classification (" + str(float(k)/num_chunks) + "% chance) avg = " + str(FT_classification_score) 
		print "fMRI -> Text (Procrustes) scene ranking (50% chance) avg = " + str(FT_rank_score) 
		print "---------------------------------------------------------------------------"
		print "Text -> fMRI (Procrustes) scene classification (" + str(float(k)/num_chunks) + "% chance) avg = " + str(TF_classification_score) 
		print "Text -> fMRI (Procrustes) scene ranking (50% chance) avg = " + str(TF_rank_score)
		print "---------------------------------------------------------------------------"
		print "///////////////////////////////////////////////////////////////////////////"
		
		mask_results[mask] = (C_weights, TF_classification_score, TF_rank_score, FT_classification_score, FT_rank_score)
	return mask_results




if __name__ == "__main__":

	results_dict = {} 
	avg_options = ['srm', 'srmica', 'pca']  #'avg': I think we can get rid of avg and just use PCA since it performs similarly anyways, in other figures / experiments too
	weight_options = ['weighted', 'unweighted']
	subtract_options = ['subtract', 'no_subtract']
	fmri_vs_text_C_opts = ['fmri', 'text'] # do we learn a weighted combination C on the fMRI side or on the text side?
	weight_list = gen_exp_weightings()

	for avg in avg_options:
		for weight in weight_options:
			for subtract in subtract_options:
				for time_weighting in weight_list:
					# test weighted combinations on both fMRI and text sides
					for fmri_vs_text_opt in fmri_vs_text_C_opts:
						exp_param_w, time_weights = time_weighting
						key_tuple = (fmri_vs_text_opt, avg, weight, subtract, exp_param_w) # weighting the semantic vectors
						print "----------------------"
						print key_tuple
						results_dict[key_tuple] = semantic_time_weighting_experiment_50_50_split(fmri_vs_text_opt, time_weights, False, avg, 'sep12' + weight, (subtract is 'subtract'))
						# save after each iteration
						p.dump(results_dict, open(PATH_TO_RESULTS + 'RESULTS/semantic_time_weighting_results_dict_top1_feb22_EXP_TIME_WEIGHTS_for_fMRI_and_Text_NEUROIMAGE.p', 'wb'))
					
	
	


