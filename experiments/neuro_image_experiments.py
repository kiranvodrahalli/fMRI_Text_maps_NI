import numpy as np

from path_to_RESULTS import PATH_TO_RESULTS
from linear_maps import learn_linear_maps, add_prev_time_steps, keep_only_last_time_step, learn_linear_combinations_of_vecs
from load_fMRI_data import load_avg, load_avg_pca, load_srm, load_srmica
from load_text_data import subtract_column_mean, load_may15_annotation_vecs, load_sep12_weighted_word_vecs_annotations, load_sep12_unweighted_word_vecs_annotations, load_sep12_sparse_weighted_word_vecs_annotations, load_sep12_sparse_unweighted_word_vecs_annotations
from load_hmm_changepoints import changepoint_dict
from create_semantic_vectors import make_weighted_average_word_vecs_over_multiple_TRs
from performance_metrics import scene_classification, scene_ranking, voting_scene_classification, hmm_scene_classification, hmm_scene_ranking

from random import shuffle
from scipy.stats import sem

import pickle as p

# avg_variant = 'avg', 'pca'
# if sparsity = 0, make sure date_of_wordvecs corresponds with a non-sparsity measure
# subtract_temporal_average_bool = pre-process the word vectors by subtracting out the temporal average
# if True
# USE THIS ONE: allows for avg, pca, srm too
def experiment_50_50_split(avg_variant, date_of_wordvecs, sparsity, subtract_temporal_average_bool):
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
	elif date_of_wordvecs == 'sep12sparse_weighted':
		semantic_vecs = load_sep12_sparse_weighted_word_vecs_annotations(sparsity)
	elif date_of_wordvecs == 'sep12sparse_unweighted':
		semantic_vecs = load_sep12_sparse_unweighted_word_vecs_annotations(sparsity)
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
		#print "Training linear maps..."
		Wridge_ft, Wridge_tf, Wpro_ft, Wpro_tf = learn_linear_maps(xrange(num_sem_vecs/2+1), [], fmri_tr, mask + avg_variant, word_tr, date_of_wordvecs)
		#print "Done training linear maps..."

		
		
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
		# comparisons in fMRI space (i.e. text -> fMRI) (best was ridge, use that map first)
		# truth is fmri_tst
		# prediction is Wridge_tf*word_tst
		# flipped is using procrustes instead
		TF_truth = fmri_tst
		TF_prediction = np.dot(Wridge_tf, word_tst)
		TF_prediction_flipped = np.dot(Wpro_tf, word_tst)
		TF_prediction = TF_prediction[0, :, :]
		TF_prediction_flipped = TF_prediction_flipped[0, :, :]
		#print "TF pred shape: " + str(TF_prediction.shape)
		#print "TF pred flipped shape: " + str(TF_prediction_flipped.shape)
		#print "TF truth shape: " + str(TF_truth.shape)
		assert(TF_truth.shape == TF_prediction.shape)
		assert(TF_truth.shape == TF_prediction_flipped.shape)
			
		TF_classification_score = scene_classification(TF_truth, TF_prediction, chunk_array, k)
		TF_classification_score_flipped = scene_classification(TF_truth, TF_prediction_flipped, chunk_array, k)
		TF_rank_score, TF_ranks = scene_ranking(TF_truth, TF_prediction, chunk_array)
		TF_rank_score_flipped, TF_ranks_flipped = scene_ranking(TF_truth, TF_prediction_flipped, chunk_array)
		TF_rank_score = TF_rank_score/(num_chunks + 0.)
		TF_rank_score_flipped = TF_rank_score_flipped/(num_chunks + 0.)

		
		# comparisons in Semantic space (i.e. fMRI -> text) (best was procrustes, use that map first)
		# truth is word_tst
		# prediction is Wpro_ft*fmri_tst
		# flipped is using ridge instead
		FT_truth = word_tst
		FT_prediction = np.dot(Wpro_ft, fmri_tst)
		FT_prediction_flipped = np.dot(Wridge_ft, fmri_tst)
		FT_prediction = FT_prediction[0, :, :]
		FT_prediction_flipped = FT_prediction_flipped[0, :, :]
		#print "FT pred shape: " + str(FT_prediction.shape)
		#print "FT pred flipped shape: " + str(FT_prediction_flipped.shape)
		#print "FT truth shape: " + str(FT_truth.shape)
		assert(FT_truth.shape == FT_prediction.shape)
		assert(FT_truth.shape == FT_prediction_flipped.shape)

		FT_classification_score = scene_classification(FT_truth, FT_prediction, chunk_array, k)
		FT_classification_score_flipped = scene_classification(FT_truth, FT_prediction_flipped, chunk_array, k)
		FT_rank_score, FT_ranks = scene_ranking(FT_truth, FT_prediction, chunk_array)
		FT_rank_score_flipped, FT_ranks_flipped = scene_ranking(FT_truth, FT_prediction_flipped, chunk_array)
		FT_rank_score = FT_rank_score/(num_chunks + 0.)
		FT_rank_score_flipped = FT_rank_score_flipped/(num_chunks + 0.)

		
		print "Mask: " + mask
		print "Using Ridge for Text -> fMRI and Procrustes for fMRI -> Text"
		print "Text -> fMRI (Ridge) scene classification (" + str(float(k)/num_chunks) + "% chance) avg = " + str(TF_classification_score) 
		print "Text -> fMRI (Ridge) scene ranking (50% chance) avg = " + str(TF_rank_score)
		print "fMRI -> Text (Procrustes) scene classification (" + str(float(k)/num_chunks) + "% chance) avg = " + str(FT_classification_score) 
		print "fMRI -> Text (Procrustes) scene ranking (50% chance) avg = " + str(FT_rank_score) 
		print "---------------------------------------------------------------------------"
		print "Using Procrustes for Text -> fMRI and Ridge for fMRI -> Text"
		print "Text -> fMRI (Procrustes) scene classification (" + str(float(k)/num_chunks) + "% chance) avg = " + str(TF_classification_score_flipped) 
		print "Text -> fMRI (Procrustes) scene ranking (50% chance) avg = " + str(TF_rank_score_flipped)
		print "fMRI -> Text (Ridge) scene classification (" + str(float(k)/num_chunks) + "% chance) avg = " + str(FT_classification_score_flipped) 
		print "fMRI -> Text (Ridge) scene ranking (50% chance) avg = " + str(FT_rank_score_flipped) 
		print "///////////////////////////////////////////////////////////////////////////"
		
		mask_results[mask] = (TF_classification_score, TF_rank_score, FT_classification_score, FT_rank_score, TF_classification_score_flipped, TF_rank_score_flipped, FT_classification_score_flipped, FT_rank_score_flipped)
	return mask_results


# uses past num_previous time steps in both learning and testing
# avg_variant = 'pca', 'srm', 'srmica'
# if sparsity = 0, make sure date_of_wordvecs corresponds with a non-sparsity measure
# subtract_temporal_average_bool = pre-process the word vectors by subtracting out the temporal average
# if True
# improper_classification = True means we feed the whole prev_time_step_extended vector into the scene classification and scene ranking
# improper_classification = False means we learn linear maps in the higher dimensional spaces, but then truncate
# improper_classification = None means we instead learn linear maps from high_dim_fmri -> no prev_time text and high_dim_text -> no prev_time fmri
# if ic=False doesn't work, need to try fitting directly from prev_stacked_fmri -> curr_text (ic=None) (and vice versa)
# can trim (the side youre mapping to) down to size before learning linear map, or after
# before = improper_classification="None"
# after = improper_classification="False"
# USE THIS ONE: allows for pca, srm, srmica
def past_timesteps_experiment_50_50_split(improper_classification, num_previous, avg_variant, date_of_wordvecs, sparsity, subtract_temporal_average_bool):
	if avg_variant == 'avg':
		print "AVG version is not supported for previous timestep mode"
		return
	elif avg_variant == 'pca':
		fmris = load_avg_pca(1)
		for mask in fmris.keys():
			fmri = fmris[mask]
			print "size of fmri PCA: " + str(fmri.shape)
			# assume that time = #cols
			num_time_steps = fmri.shape[1]
			fmri_tr = fmri[:, xrange(num_time_steps/2+1)]
			fmri_tst = fmri[:, xrange(num_time_steps/2+1, num_time_steps)]
			fmri_tr = add_prev_time_steps(fmri_tr, num_previous, range(fmri_tr.shape[1]))
			fmri_tst = add_prev_time_steps(fmri_tst, num_previous, range(fmri_tst.shape[1]))
			new_fmri = np.c_[fmri_tr, fmri_tst]
			fmris[mask] = new_fmri

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
			# add previous time steps
			# note that we add previous time steps SEPARATELY to training and test
			# this means that there is NO OVERLAP between training and test in the previous time steps.
			S_prev_times = add_prev_time_steps(S, num_previous, range(S.shape[1]))
			avg_srm_prev_times = add_prev_time_steps(avg_srm_tst_data, num_previous, range(avg_srm_tst_data.shape[1]))
			#print "shape of S_prev_times = " + str(S_prev_times.shape)
			#print "shape of avg_srm_prev_times = " + str(avg_srm_prev_times.shape)
			fmri_mask = np.c_[S_prev_times, avg_srm_prev_times]
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
	elif date_of_wordvecs == 'sep12sparse_weighted':
		semantic_vecs = load_sep12_sparse_weighted_word_vecs_annotations(sparsity)
	elif date_of_wordvecs == 'sep12sparse_unweighted':
		semantic_vecs = load_sep12_sparse_unweighted_word_vecs_annotations(sparsity)
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

	# add previous time steps to semantic stuff
	word_tr = add_prev_time_steps(word_tr, num_previous, range(word_tr.shape[1]))
	word_tst = add_prev_time_steps(word_tst, num_previous, range(word_tst.shape[1]))

	mask_results = {}
	for mask in fmris.keys():
		fmri = fmris[mask] 

		num_total_time_steps = fmri.shape[1]
		#print "fmri.shape = " + str(fmri.shape)
		fmri_tr = fmri[:, xrange(num_sem_vecs/2+1)]
		fmri_tst = fmri[:, xrange(num_sem_vecs/2 + 1, num_sem_vecs)]
		print "fmri_tr.shape = " + str(fmri_tr.shape)
		print "fmri_tst.shape = " + str(fmri_tst.shape)
		print "Training linear maps..."
		if improper_classification == None:
			# truncate one side before learning linear map (for each direction)
			# modify fmri_tr, word_tr (aren't used again)
			long_fmri_tr = fmri_tr.copy()
			long_word_tr = word_tr.copy()
			short_fmri_tr = keep_only_last_time_step(fmri_tr, num_previous, range(fmri_tr.shape[1]))
			short_word_tr = keep_only_last_time_step(word_tr, num_previous, range(word_tr.shape[1]))
			# learn long fmri -> short text
			Wridge_ft, useless1, Wpro_ft, useless2 = learn_linear_maps(xrange(num_sem_vecs/2+1), [], long_fmri_tr, mask + avg_variant, short_word_tr, date_of_wordvecs)
			# learn long text -> short fmri
			useless1, Wridge_tf, useless2, Wpro_tf = learn_linear_maps(xrange(num_sem_vecs/2+1), [], short_fmri_tr, mask + avg_variant, long_word_tr, date_of_wordvecs)
			#clear memory
			useless1 = None
			useless2 = None
			fmri_tr = None
			word_tr = None
			long_fmri_tr = None
			long_word_tr = None
			short_fmri_tr = None
			short_word_tr = None

		else:
			Wridge_ft, Wridge_tf, Wpro_ft, Wpro_tf = learn_linear_maps(xrange(num_sem_vecs/2+1), [], fmri_tr, mask + avg_variant, word_tr, date_of_wordvecs)
		print "Done training linear maps..."


		
		TF_class_scores = []
		TF_rank_scores = []
		FT_class_scores = []
		FT_rank_scores = []

		num_time_steps = fmri_tst.shape[1] # assume time steps are # of columns
		chunk_array = []
		index = 0
		num_chunks = 25 # CHANGED JAN 11
		while index < num_time_steps:
			chunk_array.append(index)
			index += num_time_steps/num_chunks
			if index > num_time_steps:
				index = num_time_steps
		k = 1 #-> 4% chance rate



		if improper_classification == True or improper_classification == False:
			# comparisons in fMRI space (i.e. text -> fMRI) (best was ridge, use that map first)
			# truth is fmri_tst
			# prediction is Wridge_tf*word_tst
			# flipped is using procrustes instead
			TF_prediction = np.dot(Wridge_tf, word_tst)
			TF_prediction_flipped = np.dot(Wpro_tf, word_tst)
			# comparisons in Semantic space (i.e. fMRI -> text) (best was procrustes, use that map first)
			# truth is word_tst
			# prediction is Wpro_ft*fmri_tst
			# flipped is using ridge instead
			FT_prediction = np.dot(Wpro_ft, fmri_tst)
			FT_prediction_flipped = np.dot(Wridge_ft, fmri_tst)


		elif improper_classification == None:
			long_fmri_tst = fmri_tst
			long_word_tst = word_tst

			# these linear maps will already be in the short space
			# so the predictions don't need to be truncated, they're already the right size
			TF_prediction = np.dot(Wridge_tf, long_word_tst)
			TF_prediction_flipped = np.dot(Wpro_tf, long_word_tst)

			FT_prediction = np.dot(Wpro_ft, long_fmri_tst)
			FT_prediction_flipped = np.dot(Wridge_ft, long_fmri_tst)

			TF_truth = keep_only_last_time_step(fmri_tst, num_previous, range(fmri_tst.shape[1]))
			FT_truth = keep_only_last_time_step(word_tst, num_previous, range(word_tst.shape[1]))

		
		Wridge_tf = None
		Wpro_tf = None
		Wpro_ft = None
		Wridge_ft = None

		if improper_classification == True:
			# no truncation
			TF_truth = fmri_tst.copy()
			FT_truth = word_tst.copy()

		# Truncate after learning linear maps btwn long and long
		elif improper_classification == False:
			# evaluate in fMRI space
			TF_prediction = keep_only_last_time_step(TF_prediction, num_previous, range(TF_prediction.shape[1]))
			TF_prediction_flipped = keep_only_last_time_step(TF_prediction_flipped, num_previous, range(TF_prediction_flipped.shape[1]))

			# evaluate in text space
			FT_prediction = keep_only_last_time_step(FT_prediction, num_previous, range(FT_prediction.shape[1]))
			FT_prediction_flipped = keep_only_last_time_step(FT_prediction_flipped, num_previous, range(FT_prediction_flipped.shape[1]))

			TF_truth = keep_only_last_time_step(fmri_tst, num_previous, range(fmri_tst.shape[1]))
			FT_truth = keep_only_last_time_step(word_tst, num_previous, range(word_tst.shape[1]))

		fmri_tst = None
		word_tst = None


		# check everything is the right size
		print "TF_prediction.shape = " + str(TF_prediction.shape)
		print "TF_prediction_flipped.shape = " + str(TF_prediction_flipped.shape)
		#print "TF pred shape: " + str(TF_prediction.shape)
		#print "TF pred flipped shape: " + str(TF_prediction_flipped.shape)
		#print "TF truth shape: " + str(TF_truth.shape)
		assert(TF_truth.shape == TF_prediction.shape)
		assert(TF_truth.shape == TF_prediction_flipped.shape)
		
		print "FT_prediction.shape = " + str(FT_prediction.shape)
		print "FT_prediction_flipped.shape = " + str(FT_prediction_flipped.shape)
		#print "FT pred shape: " + str(FT_prediction.shape)
		#print "FT pred flipped shape: " + str(FT_prediction_flipped.shape)
		#print "FT truth shape: " + str(FT_truth.shape)
		assert(FT_truth.shape == FT_prediction.shape)
		assert(FT_truth.shape == FT_prediction_flipped.shape)





		# CALCULATE SCORES
			
		TF_classification_score = scene_classification(TF_truth, TF_prediction, chunk_array, k)
		TF_classification_score_flipped = scene_classification(TF_truth, TF_prediction_flipped, chunk_array, k)
		TF_rank_score, TF_ranks = scene_ranking(TF_truth, TF_prediction, chunk_array)
		TF_rank_score_flipped, TF_ranks_flipped = scene_ranking(TF_truth, TF_prediction_flipped, chunk_array)
		TF_rank_score = TF_rank_score/(num_chunks + 0.)
		TF_rank_score_flipped = TF_rank_score_flipped/(num_chunks + 0.)

		FT_classification_score = scene_classification(FT_truth, FT_prediction, chunk_array, k)
		FT_classification_score_flipped = scene_classification(FT_truth, FT_prediction_flipped, chunk_array, k)
		FT_rank_score, FT_ranks = scene_ranking(FT_truth, FT_prediction, chunk_array)
		FT_rank_score_flipped, FT_ranks_flipped = scene_ranking(FT_truth, FT_prediction_flipped, chunk_array)
		FT_rank_score = FT_rank_score/(num_chunks + 0.)
		FT_rank_score_flipped = FT_rank_score_flipped/(num_chunks + 0.)


		# DISPLAY SCORES

		
		print "Mask: " + mask
		print "Using Ridge for Text -> fMRI and Procrustes for fMRI -> Text"
		print "Text -> fMRI (Ridge) scene classification (" + str(float(k)*100/num_chunks) + "% chance) avg = " + str(TF_classification_score) 
		print "Text -> fMRI (Ridge) scene ranking (50% chance) avg = " + str(TF_rank_score)
		print "fMRI -> Text (Procrustes) scene classification (" + str(float(k)*100/num_chunks) + "% chance) avg = " + str(FT_classification_score) 
		print "fMRI -> Text (Procrustes) scene ranking (50% chance) avg = " + str(FT_rank_score) 
		print "---------------------------------------------------------------------------"
		print "Using Procrustes for Text -> fMRI and Ridge for fMRI -> Text"
		print "Text -> fMRI (Procrustes) scene classification (" + str(float(k)*100/num_chunks) + "% chance) avg = " + str(TF_classification_score_flipped) 
		print "Text -> fMRI (Procrustes) scene ranking (50% chance) avg = " + str(TF_rank_score_flipped)
		print "fMRI -> Text (Ridge) scene classification (" + str(float(k)*100/num_chunks) + "% chance) avg = " + str(FT_classification_score_flipped) 
		print "fMRI -> Text (Ridge) scene ranking (50% chance) avg = " + str(FT_rank_score_flipped) 
		print "///////////////////////////////////////////////////////////////////////////"
		
		mask_results[mask] = (TF_classification_score, TF_rank_score, FT_classification_score, FT_rank_score, TF_classification_score_flipped, TF_rank_score_flipped, FT_classification_score_flipped, FT_rank_score_flipped)
	return mask_results





## ADD THE FOLLOWING EXPERIMENTS #
# a) fMRI/text reconstruction performance (to compare to Gallant)
# b) COMPARE WITH SKIPTHOUGHTS VECTORS!!!!!! (ideally also with TFCDL, but that doesn't work on their own dataset yet so it'll have to wait)
#    (also, if outperforms skipthoughts, get to contrast with simplicity of our approach)
# c) actual output of decoded words based on the predicted vectors
# d) (to go in a different python file perhaps): assess the constructed semantic annotation vectors 
#    (i.e., which atoms/words are nearby for a given description?)
# Already created figures which demonstrate the correlation of the constructed semantic vectors
# e) maybe make some plots demonstrating correlation of the PREDICTED semantic vectors (from fMRI)

# comparisons handled so far: 
# 1) the weighting matters (vs. unweighted), 
# 2) sparsity only helps interpretability,
# 3) SRM is better than averaging/PCA, 
# 4) Procrustes is better than Ridge
# 5) DMN-A does way better than all other regions, very clear demonstration predictively
# 6) and also binning over time (without any HMM) is no good -- may need to redo this
# 7) Low-dimensional SRM performs the best (20 dims are sufficient)
# 8) Temporal subtraction works a lot better for fMRI -> Text, and the same from Text -> fMRI
# 9) to check: divide by standard dev in each coordinate as well? 
# 10) to check: different masks 
if __name__ == "__main__":

	results_dict_truncate = {} # improper_classification = False
	results_dict_single_sided = {} # improper_classification = None
	avg_options = ['srm', 'srmica', 'avg', 'pca'] 
	weight_options = ['weighted', 'unweighted']
	subtract_options = ['subtract', 'no_subtract']
	num_previous_timesteps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30] # num_previous_timesteps 



	# load previous prev_times_dict so that we may add onto it
	# prev_times_dict = p.load(open(PATH_TO_RESULTS + 'RESULTS/prev_times_dict_top1_feb8_most_detailed.p', 'r'))

	for avg in avg_options:
		for weight in weight_options:
			for sparsity in ['']: #sparsity_options
				sparse_val = 0
				for subtract in subtract_options:
					key_tuple = (avg, weight, subtract, 0)
					print "----------------------"
					print key_tuple
					results_dict[key_tuple] = experiment_50_50_split(avg, 'sep12' + sparsity + weight, sparse_val, (subtract is 'subtract'))
					# save after each iteration
					p.dump(results_dict, open(PATH_TO_RESULTS + 'RESULTS/main_results_dict_top1_feb8_NEUROIMAGE.p', 'wb'))
					if avg != 'avg':
						# don't do previous time steps without dimension reduction
						print "PREV TIME STEPS NOW"
						for num_previous in num_previous_timesteps: # all
							key_tuple = (avg, weight, subtract, num_previous)
							print key_tuple
							print "=============================="
							# improper classification = False
							results_dict_truncate[key_tuple] = past_timesteps_experiment_50_50_split(False, num_previous, avg, 'sep12' + sparsity + weight, sparse_val, (subtract is 'subtract'))
							p.dump(results_dict_truncate, open(PATH_TO_RESULTS + 'RESULTS/main_results_dict_top1_feb11_TRUNCATE_NEUROIMAGE.p', 'wb'))
							results_dict_single_sided[key_tuple] = past_timesteps_experiment_50_50_split(None, num_previous, avg, 'sep12' + sparsity + weight, sparse_val, (subtract is 'subtract'))
							p.dump(results_dict_single_sided, open(PATH_TO_RESULTS + 'RESULTS/main_results_dict_top1_feb11_SINGLE_SIDED_NEUROIMAGE.p', 'wb'))
							print "//////////////////////"
	
	


