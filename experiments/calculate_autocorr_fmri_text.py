import numpy as np
from path_to_RESULTS import PATH_TO_RESULTS
from load_text_data import load_sep12_weighted_word_vecs_annotations, subtract_column_mean
from load_fMRI_data import load_srmica, load_srm
from performance_metrics import pearson_r, cosdist
import matplotlib.pyplot as plt

# pearson_r basically subtracts the mean out, so you could technically use it without calculating 
# the subtracted annotation mean. 
# what this means is that cosine distance for mean-subtracted vectors = pearson_r for either mean-subtracted 
# or not mean-subtracted vectors 

calculated_fmri_text_correlations = True
if not calculated_fmri_text_correlations:	
	# subtract mean
	sub_mean_annotation_vecs, avg_vec = subtract_column_mean(load_sep12_weighted_word_vecs_annotations())
	#print "avg vec= "+  str(avg_vec)
	#print "avg vec norm = " + str(np.linalg.norm(avg_vec))	

	annotation_vecs = sub_mean_annotation_vecs	

	print "Annotation vector shape: " + str(annotation_vecs.shape)

	#SRM = load_srm(20, 0)
	SRMICA = load_srmica(20, 0)
	fMRI_vecs = SRMICA["erez_dmna_network"]
	Ws = fMRI_vecs[0]
	S = fMRI_vecs[1]
	tst = fMRI_vecs[2]
	for subj in range(16):
		if subj == 0:
			S_tst = np.dot(Ws[subj, :, :].T, tst[subj, :, :]) / 16.
		else:
			S_tst += np.dot(Ws[subj, :, :].T, tst[subj, :, :]) / 16.
	fMRI_vecs = np.c_[S, S_tst]
	print "fMRI shape: " + str(fMRI_vecs.shape)	
	

	# calculate the fMRI and text chunks used in actual experiments
	# so that we can compare those correlations.	
	'''
	print "Calculating correlation between chunks"
	num_time_steps = fMRI_vecs.shape[1]
	chunk_array = []
	index = 0
	num_chunks = 50 
	while index < num_time_steps:
		chunk_array.append(index)
		index += num_time_steps/num_chunks
		if index > num_time_steps:
			index = num_time_steps	

	fMRI_chunks = []
	annotation_chunks = []
	for i in xrange(len(chunk_array)-1):
		start = chunk_array[i]
		end = chunk_array[i+1]
		curr_fchunk = fMRI_vecs[:, start:end]
		curr_achunk = annotation_vecs[:, start:end]
		curr_fchunk = np.reshape(curr_fchunk, (curr_fchunk.shape[0]*curr_fchunk.shape[1]))
		#print "fMRI chunk shape = " + str(curr_fchunk.shape)
		curr_achunk = np.reshape(curr_achunk, (curr_achunk.shape[0]*curr_achunk.shape[1]))
		#print "annotation chunk shape = " + str(curr_achunk.shape)
		fMRI_chunks.append(curr_fchunk)
		annotation_chunks.append(curr_achunk)	

	fMRI_vecs = np.matrix(fMRI_chunks).T
	print "fMRI_vecs.shape = " + str(fMRI_vecs.shape)
	annotation_vecs = np.matrix(annotation_chunks).T
	print "annotation_vecs.shape = " + str(annotation_vecs.shape)
	'''

# distance_apart = number of timesteps apart
# that we calculate correlation between
# i.e. if = 3 TRs, then you calculate sequential correlations
# between time steps 0, 3; 1, 4; 2, 5; etc. 
# setting this value to 1 means you look at correlations between adjacent timepoints
def calculate_correlations(vecs, distance_apart):
	correlations = []
	#cosdists = []
	dim = vecs.shape[0]
	num_vecs = vecs.shape[1]
	assert(distance_apart < num_vecs)
	prev_ind = 0
	for i in range(distance_apart, num_vecs):
		prev_vec = vecs[:, prev_ind]
		curr_vec = vecs[:, i]
		curr_corr = pearson_r(prev_vec, curr_vec)[0]
		#curr_cosdist = cosdist(prev_vec, curr_vec)
		correlations.append(curr_corr)
		#cosdists.append(curr_cosdist)
		prev_ind += 1
	print "Average correlation: " + str(np.mean(np.array(correlations)))
	#print "Average cos dist: " + str(np.mean(np.array(cosdists)))
	return correlations

def calculate_distances(vecs, distance_apart):
	distances = []
	dim = vecs.shape[0]
	num_vecs = vecs.shape[1]
	assert(distance_apart < num_vecs)
	prev_ind = 0
	for i in range(distance_apart, num_vecs):
		prev_vec = vecs[:, prev_ind]
		curr_vec = vecs[:, i]
		curr_dist = np.linalg.norm(prev_vec - curr_vec)
		distances.append(curr_dist)
		prev_ind += 1
	print "Average l2 distance: " + str(np.mean(np.array(distances)))
	return distances


def calculate_all_average_sequential_correlations():
	fMRI = []
	text = []
	for distance_apart in [1, 2, 3, 4, 5]:
		print "Distance apart = " + str(distance_apart)
		print "fMRI SRM-ICA 20-dim correlations"
		fMRI_corrs = calculate_correlations(fMRI_vecs, distance_apart)
		print "Annotation 100-dim correlations"
		text_corrs = calculate_correlations(annotation_vecs, distance_apart)
		fMRI.append(fMRI_corrs)
		text.append(text_corrs)
	np.savez('fMRI_and_text_single_timepoint_correlations.npz', fMRI=fMRI, text=text)
	return (fMRI, text)

def plot_correlation_time_series(chunked):
	file_str = ''
	if chunked == True:
		file_str = 'fMRI_and_text_chunked_correlations.npz'
	else:
		file_str = 'fMRI_and_text_single_timepoint_correlations.npz'
	data = np.load(file_str)
	fMRI = data['fMRI']
	text = data['text']
	for i in range(len(fMRI)):
		# distance apart = i
		fMRI_dist_i = fMRI[i]
		print "Average correlation for 20-dim fMRI SRM-ICA: " + str(np.mean(np.array(fMRI_dist_i)))
		text_dist_i = text[i]
		print "Average correlation for 100-dim annotations: " + str(np.mean(np.array(text_dist_i)))
		#print "length of fMRI corrs: " + str(len(fMRI_dist_i))
		#print "length of text corrs: " + str(len(text_dist_i))
		fig, axarr = plt.subplots() 
		axarr.plot(range(len(fMRI_dist_i)), fMRI_dist_i, label='20-dim SRM-ICA fMRI')
		axarr.plot(range(len(text_dist_i)), text_dist_i, label='100-dim annotations')
		axarr.legend(loc='lower right')
		title_str = 'Correlation over time, Distance apart = ' + str(i + 1) + ' TRs'
		if chunked == True:
			title_str = 'Chunked ' + title_str
		else:
			title_str = 'Single Timepoint ' + title_str
		axarr.set_title(title_str)
		fig_name = 'auto_corr_fig_dist_'+ str(i + 1)
		if chunked:
			fig_name += '_chunked'
		else:
			fig_name += '_single_timepoint'
		fig_name += '.eps'
		plt.savefig('/Users/kiranv/home/spr17/neuro_image_SUBMISSION_3_13_17/auto_correlation_figs/' + fig_name, format='eps', dpi=200)
		plt.show()


if __name__ == '__main__':
	if not calculated_fmri_text_correlations:
		calculate_all_average_sequential_correlations()
	plot_correlation_time_series(True)
	plot_correlation_time_series(False)

