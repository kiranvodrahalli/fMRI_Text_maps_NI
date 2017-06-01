from path_to_RESULTS import PATH_TO_RESULTS
import pickle as p
import numpy as np
from numpy import nonzero, zeros, ones, any, array, savez, dot
from numpy.linalg import norm
from scipy.io import savemat, loadmat
import operator
from sklearn.metrics.pairwise import pairwise_distances
from load_text_data import subtract_column_mean, load_sep12_weighted_word_vecs_annotations, load_sep12_unweighted_word_vecs_annotations

# some bug in here??

# VERY VERY VERY IMPORTANT CODE
# MOST RECENT: Sep 18, 2016 11:45 AM

#main_path = '/jukebox/norman/knv/thesis/sherlock/RESULTS/'
main_path = PATH_TO_RESULTS + 'RESULTS/'

# probably want to use a dict which has enwiki sub1 3sparse and 582 atoms; all 68000 or so words from enwiki
# DO NOT CHANGE THIS VALUE!!!! (unless you want to change it to a sparser value)
dict_atoms_vecs = main_path + 'SEMVECS/jl_dim_enwiki_subtract1_3sparse_477atoms_vecs.npz'

# which atoms to use, selected by hand out of the 1550 (and using the same ordering! this is important)
# DO NOT CHANGE (We included this particular file in the file above)
# all_relevant_atoms = '/Users/kiranv/home/thesis/sherlock/vec_atoms_to_use/all_atoms_to_keep.p'


da = np.load(dict_atoms_vecs)
# numwords = 1990
# numatoms = 1550
# numdims = 300
# corrwords calc'd = 100
D_300 = da['D_300'] # 300 x 1550
corr = da['corr'] # 100 x 1550
corr_words = da['corr_words'] # 100 x 1550
representation = da['representation'] # 1550 x numwords
enwiki_words = da['words'] # numwords x 1
wv_300 = da['wv_300'] # 300 x numwords
wv_100 = da['wv_100'] # 100 x numwords # TO USE -- just run rand-walk with 100 dim as parameter on wikipedia. no other processing.
wv_50 = da['wv_50'] # 50 x numwords
wv_20 = da['wv_20'] # 20 x numwords
atoms = da['atoms'] # 477 atom restriction
D_100 = da['D_100']
D_50 = da['D_50']
D_20 = da['D_20']

'''
# let's identify the atoms which correlate too highly
cos_dists = pairwise_distances(D_100.T, metric='cosine')
print cos_dists.shape # 1550 x 1550 
cscores = np.sum(cos_dists, axis=0) # sum every row
# remove atoms which correlate too highly with everything / average
def filter_by_correlation(threshold):
	good_atoms = []
	for atom_index in range(1550):
		if cscores[atom_index] <= threshold:
			good_atoms.append(atom_index) #keep
	return good_atoms

# 1250 is good based on looking at outliers from plot: 
correlation_filtered_atoms = filter_by_correlation(1250)
'''

#word_index_dict = dict()
#for i in range(enwiki_words.shape[0]):
#	w = enwiki_words[i][0][0]
#	word_index_dict[w] = i
#p.dump(word_index_dict, open('word_index_dict.p', 'wb'))

word_index_dict = p.load(open(main_path + 'SEMVECS/word_index_dict.p', 'rb'))

# get index of a word
def get_word_index(word):
	if word not in word_index_dict:
		return -1
	return word_index_dict[word]

# only dealing with 100-dimensional ones
# should return a 100 x 1976 matrix (the 3-sparse representations of calculated
# weighted/unweighted sums of word vectors)
# sparsity = 3, 5, 10 (# of atoms per vector)
def get_sparse_rep_word_vectors(weighted_bool, sparsity):
	d = dict()
	if weighted_bool == True:
		# yes weighted
		d = loadmat(main_path + 'SEMVECS/weighted_' + str(sparsity) + 'sparse_reps_of_annotations.mat')
		d = d['weighted']
	else:
		# unweighted
		d = loadmat(main_path + 'SEMVECS/unweighted_' + str(sparsity) + 'sparse_reps_of_annotations.mat')
		d = d['unweighted']

	# D_100 is 100 x 1550, dictionary rep is 1550 x 1976
	sparse_rep = dot(D_100, d)
	assert(sparse_rep.shape[0] == 100)
	assert(sparse_rep.shape[1] == 1976)
	if weighted_bool == True:
		savez(main_path + 'SEMVECS/' + str(sparsity) + 'sparse_avg_100dim_wordvec_mat_Sep12_weighted.npz', sparse_rep)
	else:
		savez(main_path + 'SEMVECS/' + str(sparsity) + 'sparse_avg_100dim_wordvec_mat_Sep12_unweighted.npz', sparse_rep)
	return sparse_rep

def get_word_vector(word):
	index = get_word_index(word)
	if index >= 0:
		return index, wv_100[:, get_word_index(word)]
	else:
		#print "word doesn't exist"
		return index, None

###############################
# LOAD SHERLOCK TEXT INFO
sherlock_TRs = main_path + 'SEMVECS/sherlock_text_TRs.p'

txt_TRs = p.load(open(sherlock_TRs, 'rb'))

# ^
# should technically strip the last three terms of txt_TRs right here
# but then would need to fix load_text so that after we re-run make_weighted_average_word_vecs
# for everything, it won't do the stripping the last 3 again.


# read the vocab count file to come up with
# weighting by yingyu
def read_vocab_freqs_into_dict():
	freq_dict = dict()
	total_count = 0
	with open(main_path + 'SEMVECS/enwiki_vocab.txt', 'rb') as f:
		for line in f:
			word, count = line.split(" ")
			freq_dict[word] = int(count)
			total_count += int(count)
	for key in freq_dict:
		freq_dict[key] /= (0. + total_count)
	p.dump(freq_dict, open('enwiki_freq_dict.p', 'wb'))
	return freq_dict




# average word vectors with weights as per MAP estimate using weights
# weighted_bool: if true, use yingyu weights. otherwise unweighted.
def make_weighted_average_word_vecs(weighted_bool):
	alpha = 0.0001 # as per yingyu
	avg_wordvec_per_TR = []
	freq_dict = p.load(open(main_path + 'SEMVECS/enwiki_freq_dict.p', 'rb'))
	for i in range(len(txt_TRs)):
		text_array = txt_TRs[i] #txt_TRs[max(i, 0):i+1] # an array of double-arrays: [ [[]], [[]], [[]] ]
		print "============= START OF TR " + str(i) + " ==============="
		txt_str = ''
		word_weight_disp = ''
		num_words_TR = 0
		wordvec_avg = None
		print "Length of text array =  " + str(len(text_array))
		for txt in text_array:
			#txt = txt[0] # remove the double array wrap
			for word in txt:
				num_words_TR += 1
				txt_str += word + ' '
			for word in txt:
				good_return, wordvec = get_word_vector(word)
				if good_return != -1:
					if weighted_bool:
						yingyu_coeff = alpha/(0. + freq_dict[word] + alpha)
						word_weight_disp += word + " (" + str(alpha/(0. + freq_dict[word] + alpha)) + ")  "
					else:
						yingyu_coeff = 1.
					if wordvec_avg == None:
						wordvec_avg = yingyu_coeff*wordvec/(0. + num_words_TR)
					else:
						wordvec_avg += yingyu_coeff*wordvec/(0. + num_words_TR)
				#print "-----"
		avg_wordvec_per_TR.append(wordvec_avg)
		print "Full string: " + txt_str
		print word_weight_disp
		print "============= END OF TR " + str(i) + " ==============="
	avg_word_vec_mat = np.matrix(avg_wordvec_per_TR)
	print "shape of wordvec mat = " + str(avg_word_vec_mat.shape)
	# save as file for learning the Dictionary representation
	filename = main_path + 'SEMVECS/'
	if weighted_bool:
		filename += 'weighted_'
	else:
		filename += 'unweighted_'
	filename += 'word_vector_average.mat'
	savemat(filename, {'vecs': avg_word_vec_mat})
	return avg_word_vec_mat




# average word vectors with weights as per MAP estimate using weights
# weighted_bool: if true, use yingyu weights. otherwise unweighted.
# number_TRs_we_average_over = how many TRs we calculate the annotation vector for

## IF WE IMPLEMENT A LOAD VERSION OF THIS METHOD in load_text, IT WILL HAVE TO BE DIFFERENT FROM THE OTHERS
## SINCE WE TAKE CARE OF BOTH REMOVING THE LAST THREE TRs AS WELL AS RETURNING A TRANSPOSE

## NEED TO FIX THIS - incorrect reduction in time from 1976 to 396 -> 393: we need to 
## THROW OUT 3 OF THE VECTORS BEFORE WE DO THE AVERAGING
## AS IN THE fMRI CASE! (those are all shape 1973 initially, and then we average over that)
# so need to care of that asap. then 396/2 = 198, (split into training testing afterwards), and it agrees
# subtract_out_temporal_mean_bool = True -> subtract out the mean before returning
def make_weighted_average_word_vecs_over_multiple_TRs(weighted_bool, number_TRs_we_average_over, subtract_out_temporal_mean_bool):
	alpha = 0.0001 # as per yingyu
	avg_wordvec_per_multiple_TRs = []
	freq_dict = p.load(open(main_path + 'SEMVECS/enwiki_freq_dict.p', 'rb'))
	# implement the avoidance of the last 3 TRs here
	num_TRs = len(txt_TRs) - 3
	# just ensuring that it'll never go past that TR number, which is what we want
	for i in range(0, num_TRs, number_TRs_we_average_over):
		# make sure we don't go past num_TRs
		endpoint = min(num_TRs, i + number_TRs_we_average_over)
		text_array = [txt_TRs[j][0] for j in range(i, endpoint)] 
		#print "============= START OF TR " + str(i) + " ==============="
		txt_str = ''
		num_words_TR = 0
		wordvec_avg = None
		#print "Length of text array =  " + str(len(text_array))
		#print text_array
		# calculate number of words in all the TRs as well as the text description
		for txt in text_array:
			for word in txt:
				num_words_TR += 1
				txt_str += word + ' '
		for txt in text_array:
			for word in txt:
				#print "Word: " + word
				good_return, wordvec = get_word_vector(word)
				if good_return != -1:
					if weighted_bool:
						yingyu_coeff = alpha/(0. + freq_dict[word] + alpha)
					else:
						yingyu_coeff = 1.
					if wordvec_avg == None:
						wordvec_avg = yingyu_coeff*wordvec/(0. + num_words_TR)
					else:
						wordvec_avg += yingyu_coeff*wordvec/(0. + num_words_TR)
				#print "-----"
		avg_wordvec_per_multiple_TRs.append(wordvec_avg)
		#print "============= END OF TR " + str(i) + " ==============="
	avg_word_vec_over_multiple_TRs_mat = np.matrix(avg_wordvec_per_multiple_TRs)
	avg_word_vec_over_multiple_TRs_mat = avg_word_vec_over_multiple_TRs_mat.T
	print "avg_word_vec_over_multiple_TRs_mat.shape = " + str(avg_word_vec_over_multiple_TRs_mat.shape)
	if subtract_out_temporal_mean_bool:
		avg_word_vec_over_multiple_TRs_mat = subtract_column_mean(avg_word_vec_over_multiple_TRs_mat)
		print "after subtracting: avg_word_vec_over_multiple_TRs_mat.shape = " + str(avg_word_vec_over_multiple_TRs_mat.shape)
	#print "shape of wordvec mat = " + str(avg_word_vec_over_multiple_TRs_mat.shape)
	# save as file for learning the Dictionary representation
	filename = main_path + 'SEMVECS/'
	if weighted_bool:
		filename += 'weighted_'
	else:
		filename += 'unweighted_'
	filename += 'word_vector_average_over_' + str(number_TRs_we_average_over) + '_TRs'
	if subtract_out_temporal_mean_bool:
		filename += '_subtract_out_temporal_mean'
	filename += '.mat'
	savemat(filename, {'vecs': avg_word_vec_over_multiple_TRs_mat})
	return avg_word_vec_over_multiple_TRs_mat # return transpose here to avoid flipping





# given word index, get the atoms composing it
# should also get weights
def get_atoms(word_index):
	best_atoms = dict()
	rep = representation[:, word_index]
	if norm(rep) == 0:
		return
	these_atoms = nonzero(rep)[0]
	weights = rep[these_atoms]
	tot_weight = sum(abs(weights))
	scaled_weights = weights/(1. * tot_weight) # rescale
	for i in range(these_atoms.shape[0]):
		atom_num = these_atoms[i]
		atom_weight = weights[i]
		atom_scaled_weight = scaled_weights[i]
		# cutoff! not scaling weights seems to work better (cut off less stuff)
		if abs(atom_weight) > 1.: # this ignores atoms with negative weight..
			if True: #atom_num in atoms: #correlation_filtered_atoms: #True: #atom_num in atoms: # AUTOMATE - DONT USE 477 handselected
				# for a given word, you can't have multiple of the same atom
				best_atoms[atom_num] = atom_weight
				print 'atom # ' + str(atom_num) + '; weight: ' + str(atom_weight) + "; scaled: " + str(atom_scaled_weight)
				for j in range(5):
					print corr_words[j, atom_num][0] + ": " + str(corr[j, atom_num])
				print "-------"
	return best_atoms # is a dict

# given an atom, get words that use it
def get_words(atom):
	related = []
	rep = representation[atom, :]
	if norm(rep) == 0:
		return
	rel_words = nonzero(rep)[0]
	for i in range(rel_words.shape[0]):
		index = rel_words[i]
		related.append(enwiki_words[index][0][0])
	return related


#with open("477_selected_atoms_words.txt", 'wb') as f:
#	output = ''
#	for i in range(1550):
#		if i in atoms:
#			related_words = get_words(i)
#			rel_str = ''
#			for w in related_words:
#				rel_str += w + ' '
#			output += 'atom # ' + str(i) + ": " + rel_str + '\n'
#			output += "--------\n"
#			cwi = corr_words[0:5, i]
#			for j in range(5):
#				output += cwi[j][0] + "\n"
#			output += "=============\n"
#	f.write(output)




# NEED TO EDIT: MAKE SURE EACH TR HAS MULTIPLE SENTENCES
# POLICY HERE: ONLY USE ATOMS, this is for ATOM CONTEXT CREATION!
# We add CHARACTERS separately and LATER (using the .xlsx file and more parsing)
# atoms_TRs_dense_vec = [] # this is atom geometry: ignore weights
# atoms_TRs_sparse_vec = [] # this is weighted indicators for which atoms are active
# atoms_TRs_weighted_dense_vec = [] # this is atom geometry: include weights
# atoms_TRs_single_dense_vec = [] # just a single atom
# atoms_TRs_weighted_top3_dense_vec = [] # weighted sum of top 3 atoms chosen by linear coefficient
#atoms_TRs_weighted_top4_dense_vec_50dim = [] # ^ except top 4, 50 dim
#atoms_TRs_weighted_top4_dense_vec_20dim = [] # ^ except top 4, 20 dim


'''
atoms_TRs_weighted_top4_dense_vec_100dim = [] # ^ except top 4, 100 dim

#context_list = [atoms_TRs_weighted_top4_dense_vec_20dim, atoms_TRs_weighted_top4_dense_vec_50dim, atoms_TRs_weighted_top4_dense_vec_100dim]
context_list = [atoms_TRs_weighted_top4_dense_vec_100dim]
for dim in [100]:
	TR = 0
	if dim == 20:
		Dictionary = D_20
	elif dim == 50:
		Dictionary = D_50
	else:
		Dictionary = D_100
	for i in range(len(txt_TRs)):
		# no big context, works worse than just doing one context at a time
		text_array = txt_TRs[i] #txt_TRs[max(i, 0):i+1] # an array of double-arrays: [ [[]], [[]], [[]] ]
		print TR
		top_atoms = dict()
		txt_str = ''
		num_words_TR = 0
		print "Check!: " + str(len(text_array))
		for txt in text_array:
			#txt = txt[0] # remove the double array wrap
			for word in txt:
				num_words_TR += 1
				txt_str += word + ' '
			print "Related Atoms: "
			for word in txt:
				print "Word: " + word
				index = get_word_index(word)
				if index not in atoms:
					print str(index) + " not in atoms, which has size " + str(len(atoms))
				if index >= 0: # word exists
					TR_atoms_dict = get_atoms(index)
					for a in TR_atoms_dict.keys():
						if a in top_atoms: 
							top_atoms[a] += TR_atoms_dict[a] 
						else:
							top_atoms[a] = TR_atoms_dict[a] #init
				print "===="
		print "Current TR " + str(TR) + ": " + txt_str



		# # create sparse vector (1550-dimensional, with the weight of each active atom in appropriate place)
		# sparse_vec = zeros(1550)
		# #total_weight = 0. # normalize here
		# for atom_num in top_atoms.keys():
		# 	weight = top_atoms[atom_num]
		# 	#total_weight += weight
		# 	sparse_vec[atom_num] = weight
		# #sparse_vec = sparse_vec/(total_weight*1.0)
		# if norm(sparse_vec) == 0:
		# 	print str(TR) + " sparsevec"
		# else:
		# 	sparse_vec = sparse_vec/norm(sparse_vec)
		# atoms_TRs_sparse_vec.append(sparse_vec)
		# # create dense single context vector by averaging
		# # just sum and average; that way atoms which appear twice will be more important
		# dense_vec = zeros(dim) # this ignores the weights so that we deal only with atoms
		# num_atoms = 0

# 		# for atom_num in top_atoms.keys():
		# 	curr_vec = Dictionary[:, atom_num]
		# 	dense_vec += curr_vec
		# 	num_atoms += 1
		# if norm(dense_vec) == 0:
		# 	print str(TR) + " densevec"
		# else:
		# 	dense_vec /= (1.0 * num_atoms)
		# atoms_TRs_dense_vec.append(dense_vec)	

# 		# # here, we do NOT ignore the weights! what we're doing is refining the word vectors
		# # by leaving out "unimportant atoms" - to some extent automatically (by only looking at top_atoms)
		# # so, we're 'averaging' over 'refined word vectors'
		# weighted_dense_vec = zeros(dim)
		# for atom_num in top_atoms.keys():
		# 	curr_vec = Dictionary[:, atom_num]
		# 	weight = top_atoms[atom_num]
		# 	weighted_dense_vec += weight*curr_vec
		# atoms_TRs_weighted_dense_vec.append(weighted_dense_vec/(0.0 + num_words_TR))	

# 		# # here we take only one atom (whichever atom has most weight)
		# # this seems like it shouldn't work.
		# # the correlation matrix ends up having a lot less correlation
		# single_vec = zeros(dim)
		# curr_weight = 0
		# for atom_num in top_atoms.keys():
		# 	curr_vec = Dictionary[:, atom_num]
		# 	weight = top_atoms[atom_num]
		# 	if weight > curr_weight:
		# 		single_vec = curr_vec
		# 		curr_weight = weight
		# atoms_TRs_single_dense_vec.append(single_vec)	

# 		# # instead of doing the full weighted thing for all things that pass our weighting scheme
		# # let's just take the top 3 atoms weighted average
		# # note that by sorting by weight is maybe not the best thing; we're just indicating that
		# # we're choosing atoms which are a great proportion of our vectors...
		# # how to choose top 3 correctly? This is not as fine-grained as it could be
		# top3_vec = zeros(dim)
		# # sort the dict by weight
		# sorted_atoms = sorted(top_atoms.items(), key=operator.itemgetter(1), reverse=True)
		# top3 = sorted_atoms[0:3]
		# #print "---+++++++++++++++++++---"
		# #print top3
		# #print "---+++++++++++++++++++---"
		# num_vecs = len(top3)
		# top3_vec = (1/float(num_vecs))*sum(Dictionary[:, top3[i][0]]*top3[i][1] for i in range(num_vecs))  
		# atoms_TRs_weighted_top3_dense_vec.append(top3_vec)	

		top4_vec = zeros(dim)
		# sort the dict by weight
		sorted_atoms = sorted(top_atoms.items(), key=operator.itemgetter(1), reverse=True)
		top4 = sorted_atoms[0:4]
		print "---+++++++++++++++++++---"
		print top4
		print "---+++++++++++++++++++---"
		num_vecs = len(top4)
		# take more than 4 if larger?
		top4_vec = (1/float(num_vecs))*sum(Dictionary[:, top4[i][0]]*top4[i][1] for i in range(num_vecs))
		if dim == 20:
			context_list[0].append(top4_vec)
		elif dim == 50:
			context_list[1].append(top4_vec)
		else:
			atoms_TRs_weighted_top4_dense_vec_100dim.append(top4_vec)
		TR += 1
		# hand-selection version too slow
		#for i in range(3):
		#	atomi = raw_input("Enter an Atom #: ")
		#	atomi = int(atomi)
		#	top3_atoms.append(atomi)
		#atoms_TRs.append(top3_atoms)
'''

#atoms_TRs_sparse_vec = array(atoms_TRs_sparse_vec)
#savez('atoms_TRs_1550dim_sparse_vec_pmweights.npz', atom_TRs=atoms_TRs_sparse_vec)#

#atoms_TRs_dense_vec = array(atoms_TRs_dense_vec)
#savez('atoms_TRs_300dim_dense_vec_pmweights.npz', atom_TRs=atoms_TRs_dense_vec)#

#atoms_TRs_weighted_dense_vec = array(atoms_TRs_weighted_dense_vec)
#savez('atoms_TRs_300dim_weighted_dense_vec_pmweights.npz', atom_TRs=atoms_TRs_weighted_dense_vec)#

#atoms_TRs_single_dense_vec = array(atoms_TRs_single_dense_vec)
#savez('atoms_TRs_300dim_single_dense_vec_pmweights.npz', atom_TRs=atoms_TRs_single_dense_vec)

#atoms_TRs_weighted_top3_dense_vec = array(atoms_TRs_weighted_top3_dense_vec)
#savez('atoms_TRs_300dim_weighted_top3_dense_vec_pmweights.npz', atom_TRs=atoms_TRs_weighted_top3_dense_vec)

#for i in range(3):
#	context_list[i] = array(context_list[i])

def delta_shift(delta, wsize, timecourse):
	t = timecourse # TRs x dim
	num_TRs = t.shape[0]
	dim = t.shape[1]
	smooth_t = np.zeros((num_TRs, dim))
	# trailing window of size wsize
	for i in range(num_TRs):
		print "TR # " + str(i)
		# window
		d_smooth_i = np.zeros(dim)
		furthest_back = max(0, i - wsize)
		print "window: "
		for j in range(furthest_back, i + 1):
			print "TR " + str(j)
			d_smooth_i += t[j, :]* (delta**(i - j))
		smooth_t[i, :] = d_smooth_i
	return smooth_t

def extract_dictionary_from_npz_to_mat():
	# D_100 is the 100dim dictionary we need
	savemat('/jukebox/norman/knv/thesis/RANDWALK/dictionary/learn/d100.mat', {'d': D_100})

if __name__ == '__main__':
	#savez('atoms_TRs_AUTO_correlation_filtered_20dim_weighted_top4_dense_vec_pmweights.npz', atom_TRs=context_list[0])
	#savez('atoms_TRs_AUTO_correlation_filtered_50dim_weighted_top4_dense_vec_pmweights.npz', atom_TRs=context_list[1])
	'''
	savez('atoms_TRs_AUTO_100dim_weighted_top4_dense_vec_pmweights.npz', atom_TRs=atoms_TRs_weighted_top4_dense_vec_100dim)#context_list[2])	
	'''
	#extract_dictionary_from_npz_to_mat()
	# RUN THIS ONE
	#savez('/jukebox/norman/knv/thesis/sherlock/RESULTS/SEMVECS/avg_100dim_wordvec_mat_Sep12_weighted.npz', vecs=make_weighted_average_word_vecs(True))
	#savez('/jukebox/norman/knv/thesis/sherlock/RESULTS/SEMVECS/avg_100dim_wordvec_mat_Sep12_unweighted.npz', vecs=make_weighted_average_word_vecs(False))
	
	#get_sparse_rep_word_vectors(True, 10)
	#get_sparse_rep_word_vectors(False, 10)

	weighted_avg_over_3TR_minus_temporal_mean = make_weighted_average_word_vecs_over_multiple_TRs(True, 3, True)
	savez(main_path + 'SEMVECS/weighted_wordvec_average_over_3TR_minus_temporal_mean.npz', vecs=weighted_avg_over_3TR_minus_temporal_mean)

	'''
	# subtract temporal average and save
	weighted_wv_avg = load_sep12_weighted_word_vecs_annotations()
	unweighted_wv_avg = load_sep12_unweighted_word_vecs_annotations()

	subtracted_weighted_wv_avg = subtract_column_mean(weighted_wv_avg)
	subtracted_unweighted_wv_avg = subtract_column_mean(unweighted_wv_avg)

	savez('weighted_wordvec_averages_subtract_temporal_mean_sep18.npz', vecs=subtracted_weighted_wv_avg)
	savez('unweighted_wordvec_averages_subtract_temporal_mean_sep18.npz', vecs=subtracted_unweighted_wv_avg)
	'''




	#dshift20 = delta_shift(0.8, 4, context_list[0])
	#dshift50 = delta_shift(0.8, 4, context_list[1])
	#dshift100 = delta_shift(0.8, 4, context_list[2])	

	#savez('atoms_TRs_delta0.8_window4_20dim_weighted_top4_dense_vec_pmweights.npz', atom_TRs=dshift20)
	#savez('atoms_TRs_delta0.8_window4_50dim_weighted_top4_dense_vec_pmweights.npz', atom_TRs=dshift50)
	#savez('atoms_TRs_delta0.8_window4_100dim_weighted_top4_dense_vec_pmweights.npz', atom_TRs=dshift100)

