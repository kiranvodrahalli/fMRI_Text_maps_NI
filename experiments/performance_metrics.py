import numpy as np
from numpy.linalg import norm 
from scipy import stats
from scipy.stats import pearsonr
import operator

# cosine similarity
def cosdist(x, y):
	return np.dot(np.ravel(x), np.ravel(y))/(norm(x)*norm(y))
# pearson_r
def pearson_r(x, y):
	return pearsonr(np.ravel(x), np.ravel(y))

# average a chunk over time so that we don't have shape problems
# assume chunk is features x chunk_length
# in the future, can do a weighted average depending on the time points
# i.e., where you are in the chunk/scene. 
# for instance, it might make sense to more heavily weight the END of a scene
# rather than the beginning or middle, since this is where the brain activity might be stable
def collapse_chunk(chunk):
	assert(len(chunk.shape) == 2)
	#print "chunk shape: " + str(chunk.shape)
	num_time_steps = chunk.shape[1]
	avg = None
	for i in range(num_time_steps):
		if i == 0:
			avg = chunk[:, 0] /(0.0 + num_time_steps)
		else:
			avg += chunk[:, i] /(0.0 + num_time_steps)
	return avg




# implements scene classification metric (a (typically) 20% probability task, if we use top 5)
# given predicted fMRI or text, compare with true fMRI or text in the scene classification framework
# prediction is a matrix: dimension x time points
# truth is a matrix: dimension x time points
# scene_divisions delineates what we consider to be 'chunks' or 'scenes'
# scene_divisions format: [a, b, c, d] means there are three scenes: 
#       a->b-1 inclusive (a:b)
#       b->c-1 inclusive (b:c)
#       c->d-1 inclusive (c:d)
# need to average over all scenes
# k is the "top #": if our best answer is in the top-k, we are good

## SWITCHED TRUTH AND PREDICTION - in test_map_performances code, this was the assumption
# does it change performance?? (given truth compare with all predictions vs. given prediction compare with truth)
# should not change results that much since we see that diagonal is strong (which is what we care about)
# may change ranking
def scene_classification(truth, prediction, scene_divisions, k):
	num_chunks = len(scene_divisions)-1
	assert(k <= len(scene_divisions)-1)
	chance_rate = k/(0. + num_chunks)
	#print "Chance rate: " + str(chance_rate)
	predict_chunks = []
	truth_chunks = []
	for i in xrange(len(scene_divisions)-1):
		start = scene_divisions[i]
		end = scene_divisions[i+1]
		curr_pchunk = prediction[:, start:end]
		curr_tchunk = truth[:, start:end]
		predict_chunks.append(curr_pchunk)
		truth_chunks.append(curr_tchunk)
	assert(len(predict_chunks) == num_chunks)
	assert(len(truth_chunks) == num_chunks)
	score = 0
	for p in xrange(num_chunks):
		curr_pchunk = predict_chunks[p]
		chunk_p_correlations = dict()
		for t in xrange(num_chunks):
			corr, pval = pearson_r(curr_pchunk, truth_chunks[t])
			chunk_p_correlations[t] = corr
		# NEEDS TO BE IN DECREASING ORDER!!!
		sorted_p_corr = sorted(chunk_p_correlations.items(), key=operator.itemgetter(1), reverse=True)
		topk = sorted_p_corr[0:k]
		topk = [topk[i][0] for i in xrange(k)]
		if p in topk:
			score += 1
	score = score /(0. + num_chunks)
	return score


# simplest version - just add up votes, no learning involved
# weight_by_position: if True, then aggregate votes based on the rank that each map picks
# and pick the smallest overall place - this is basically a hard threshold - set to 0 if not in top k
def voting_scene_classification(truth, predictions, scene_divisions, weight_by_position_bool, k):
	num_chunks = len(scene_divisions)-1
	assert(k <= len(scene_divisions)-1)
	chance_rate = k/(0. + num_chunks)
	#print "Chance rate: " + str(chance_rate)
	# prediction for different brain maps
	# for each map, give a list of length num_chunks for the predictions of each chunk
	diff_map_predictions = {}
	for j in range(len(predictions)):
		prediction = predictions[j]
		predict_chunks = []
		truth_chunks = []
		for i in xrange(len(scene_divisions)-1):
			start = scene_divisions[i]
			end = scene_divisions[i+1]
			curr_pchunk = prediction[:, start:end]
			curr_tchunk = truth[:, start:end]
			predict_chunks.append(curr_pchunk)
			truth_chunks.append(curr_tchunk)
		assert(len(predict_chunks) == num_chunks)
		assert(len(truth_chunks) == num_chunks)
		score = 0
		# THIS PART IS WHERE WE NEED TO FIX THINGS - 1/2/2017
		diff_map_predictions[j] = {} 
		for p in xrange(num_chunks):
			curr_pchunk = predict_chunks[p]
			chunk_p_correlations = []
			chunk_p_correlation_dict = dict()
			for t in xrange(num_chunks):
				corr, pval = pearson_r(curr_pchunk, truth_chunks[t])
				chunk_p_correlations.append(corr)
				chunk_p_correlation_dict[t] = corr
			# NEEDS TO BE IN DECREASING ORDER!!!
			sorted_p_corr = sorted(chunk_p_correlation_dict.items(), key=operator.itemgetter(1), reverse=True)
			topk = sorted_p_corr[0:k] # decreasing order by correlation, so index k - 1 contains min
			p_corr_indices = [sorted_p_corr[i][0] for i in xrange(len(sorted_p_corr))]
			# find the threshold correlation value: if < than this, not in top k
			topk_threshold_corr = topk[k -1][1]
			# don't sort  here since we don't do top k yet
			p_corr = np.array(chunk_p_correlations)
			#print "shape of p_corr: " + str(p_corr.shape)
			if not weight_by_position_bool: # binarize the correlations to be 1 if in top k, 0 o/w
				def bin_filter(x):
					if x < topk_threshold_corr:
						return 0
					else:
						return 1
				p_corr = np.array(map(lambda x: bin_filter(x), p_corr))
			else: # we weight by position. Give the correct rank
				for rank in range(len(p_corr_indices)):
					index = p_corr_indices[rank]
					p_corr[index] = rank # i.e., index 19 is the second highest correlation. it gets value 2.
			# mask j, prediction chunk p = either binary valued or ranked all 25 true chunks
			diff_map_predictions[j][p] = p_corr

	# time to sum over different brain maps/masks
	aggregate_predictions = {}
	for p in xrange(num_chunks):
		# here, we have not added any weighting for each different brain map, just plain averaging
		for j in range(len(predictions)):
			if p not in aggregate_predictions.keys():
				aggregate_predictions[p] = diff_map_predictions[j][p] / (0. + len(predictions))
			else:
				aggregate_predictions[p] += diff_map_predictions[j][p] / (0. + len(predictions))
		aggregated = {}
		for i in range(num_chunks):
			aggregated[i] = aggregate_predictions[p][i]
		aggregate_predictions[p] = aggregated


	# now we calculate the score
	score = 0
	# if not weight_by_position_bool, we are binary:
	# each brain map votes for a different chunk if it's in their top k
	# in this case, the chunk with the highest # of votes wins.
	# we evaluate by checking if the correct chunk is in the top k aggregate score
	if not weight_by_position_bool:
		for p in xrange(num_chunks):
			# correct chunk is index p, check if it's in the top k
			votes = aggregate_predictions[p]
			# NEEDS TO BE IN DECREASING ORDER!!!
			sorted_p_votes = sorted(votes.items(), key=operator.itemgetter(1), reverse=True)
			topk = sorted_p_votes[0:k]
			topk = [topk[i][0] for i in xrange(k)]
			if p in topk:
				score += 1
	else:
		# we are weight by position bool, and we've basically calculated the average rank across
		# brain masks for each of the true chunks given the predicted chunk p. 
		# Thus, we order the true chunks by SMALLEST RANK. 
		# if we are in the bottom k ranks, we record a victory.
		for p in xrange(num_chunks):
			votes = aggregate_predictions[p]
			# NEEDS TO BE IN INCREASING ORDER!!!
			sorted_p_ranks = sorted(votes.items(), key=operator.itemgetter(1), reverse=False)
			topk = sorted_p_ranks[0:k]
			topk = [topk[i][0] for i in xrange(k)]
			if p in topk:
				score += 1
	score = score /(0. + num_chunks) # average
	return score

# scenes can be different lengths now
def hmm_scene_classification(truth, prediction, scene_divisions, k):
	num_chunks = len(scene_divisions)-1
	assert(k <= len(scene_divisions)-1)
	chance_rate = k/(0. + num_chunks)
	#print "Chance rate: " + str(chance_rate)
	# we need to convert the chunks into averages (could be weighted) of the vectors over timesteps
	predict_chunks = []
	truth_chunks = []
	for i in xrange(len(scene_divisions)-1):
		start = scene_divisions[i]
		end = scene_divisions[i+1]
		curr_pchunk = prediction[:, start:end]
		curr_tchunk = truth[:, start:end]
		predict_chunks.append(curr_pchunk)
		truth_chunks.append(curr_tchunk)
	assert(len(predict_chunks) == num_chunks)
	assert(len(truth_chunks) == num_chunks)
	score = 0
	for p in xrange(num_chunks):
		curr_pchunk = predict_chunks[p]
		chunk_p_correlations = dict()
		for t in xrange(num_chunks):
			corr, pval = pearson_r(collapse_chunk(curr_pchunk), collapse_chunk(truth_chunks[t])) # HAS BEEN CHANGED FOR DIFF LENS TO WORK
			chunk_p_correlations[t] = corr
		# NEEDS TO BE IN DECREASING ORDER!!!
		sorted_p_corr = sorted(chunk_p_correlations.items(), key=operator.itemgetter(1), reverse=True)
		topk = sorted_p_corr[0:k]
		topk = [topk[i][0] for i in xrange(k)]
		if p in topk:
			score += 1
	score = score /(0. + num_chunks)
	return score

# returns avg rank and vector of predicted ranks 
# implements scene ranking experiment (a 50% probability task)
def scene_ranking(truth, prediction, scene_divisions):
	num_chunks = len(scene_divisions)-1
	chance_rate = 0.5
	#print "Chance rate: " + str(chance_rate)
	predict_chunks = []
	truth_chunks = []
	for i in xrange(len(scene_divisions)-1):
		start = scene_divisions[i]
		end = scene_divisions[i+1]
		curr_pchunk = prediction[:, start:end]
		curr_tchunk = truth[:, start:end]
		predict_chunks.append(curr_pchunk)
		truth_chunks.append(curr_tchunk)
	assert(len(predict_chunks) == num_chunks)
	assert(len(truth_chunks) == num_chunks)
	score = 0
	p_ranks = [] # rank of chunk p - can see which chunks do well and which do poorly, what the variance is...
	for p in xrange(num_chunks):
		curr_pchunk = predict_chunks[p]
		chunk_p_correlations = dict()
		for t in xrange(num_chunks):
			corr, pval = pearson_r(curr_pchunk, truth_chunks[t])
			chunk_p_correlations[t] = corr
		# NEEDS TO BE IN DECREASING ORDER!!!
		sorted_p_corr = sorted(chunk_p_correlations.items(), key=operator.itemgetter(1), reverse=True)
		for i in xrange(len(sorted_p_corr)):
			which_chunk = sorted_p_corr[i][0] # gives the id of the current chunk
			if which_chunk == p:
				p_ranks.append(i)
				score += (1 - i/(0. + num_chunks)) # higher is better now, i.e. 70% means it's in top 30%
				break
	return score, p_ranks

# for different scene division lengths
# returns avg rank and vector of predicted ranks 
# implements scene ranking experiment (a 50% probability task)
def hmm_scene_ranking(truth, prediction, scene_divisions):
	num_chunks = len(scene_divisions)-1
	chance_rate = 0.5
	#print "Chance rate: " + str(chance_rate)
	predict_chunks = []
	truth_chunks = []
	for i in xrange(len(scene_divisions)-1):
		start = scene_divisions[i]
		end = scene_divisions[i+1]
		curr_pchunk = prediction[:, start:end]
		curr_tchunk = truth[:, start:end]
		predict_chunks.append(curr_pchunk)
		truth_chunks.append(curr_tchunk)
	assert(len(predict_chunks) == num_chunks)
	assert(len(truth_chunks) == num_chunks)
	score = 0
	p_ranks = [] # rank of chunk p - can see which chunks do well and which do poorly, what the variance is...
	for p in xrange(num_chunks):
		curr_pchunk = predict_chunks[p]
		chunk_p_correlations = dict()
		for t in xrange(num_chunks):
			corr, pval = pearson_r(collapse_chunk(curr_pchunk), collapse_chunk(truth_chunks[t])) # HAS BEEN FIXED FOR DIFFERENT SCENE LENGTHS
			chunk_p_correlations[t] = corr
		# NEEDS TO BE IN DECREASING ORDER!!!
		sorted_p_corr = sorted(chunk_p_correlations.items(), key=operator.itemgetter(1), reverse=True)
		for i in xrange(len(sorted_p_corr)):
			which_chunk = sorted_p_corr[i][0] # gives the id of the current chunk
			if which_chunk == p:
				p_ranks.append(i)
				score += (1 - i/(0. + num_chunks)) # higher is better now, i.e. 70% means it's in top 30%
				break
	return score, p_ranks

# visualizing vectors as correlation matrices
def build_rank_and_corr_mats(truth, prediction, scene_divisions):
	num_chunks = len(scene_divisions)-1
	predict_chunks = []
	truth_chunks = []
	for i in xrange(len(scene_divisions)-1):
		start = scene_divisions[i]
		end = scene_divisions[i+1]
		curr_pchunk = prediction[:, start:end]
		curr_tchunk = truth[:, start:end]
		predict_chunks.append(curr_pchunk)
		truth_chunks.append(curr_tchunk)
	assert(len(predict_chunks) == num_chunks)
	assert(len(truth_chunks) == num_chunks)
	# construct matrices which have rows = all chunks; columns = testing index
	# i.e., each column k is the correlation of predicted chunk k with each of the true chunks
	# the 'correct' answer is always row k
	corr_mat = np.zeros((num_chunks, num_chunks))
	rank_mat = np.zeros((num_chunks, num_chunks))
	# first for loop (index p) is the true index
	for p in xrange(num_chunks):
		curr_pchunk = predict_chunks[p]
		chunk_p_correlations = dict()
		corr_vec_p = np.zeros(num_chunks)
		# for every random index t, we look at correlation
		for t in xrange(num_chunks):
			corr, pval = pearson_r(curr_pchunk, truth_chunks[t])
			chunk_p_correlations[t] = corr
			corr_vec_p[t] = corr
		corr_mat[:, p] = corr_vec_p
		sorted_p_corr = sorted(chunk_p_correlations.items(), key=operator.itemgetter(1), reverse=True)
		# sorted_p_indices[i] is the chunk # with rank i
		sorted_p_indices = map(lambda tup: tup[0], sorted_p_corr)
		rank_vec_p = np.zeros(num_chunks)
		# rank from 0 to 1: each column has 0, 1/25, 2/25, ..., 1
		for i in xrange(num_chunks):
			rank_vec_p[sorted_p_indices[i]] = 1. - i/(0. + num_chunks)
		rank_mat[:, p] = rank_vec_p
	return corr_mat, rank_mat

# no reconstruction, not as useful to display
# no binary, use scene ranking instead
