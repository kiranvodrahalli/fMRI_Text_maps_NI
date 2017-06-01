import numpy as np
from numpy.linalg import norm
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from scipy.io import loadmat
from sklearn.metrics.pairwise import pairwise_distances

# this path is old and probably should be different (inside of RESULTS)
# will not work on DROPBOX version
path = '/jukebox/norman/knv/thesis/sherlock/context_vecs_to_use/enwiki_subtract1_3sparse_477atoms_vecs.mat'

enwiki = loadmat(path)

words = enwiki['words'] # 1990 x 1
Dictionary = enwiki['Dictionary'] # 300 x 1550
corr = enwiki['corr'] # 100 x 1550 
corr_words = enwiki['corr_words'] # 100 x 1550
representation = enwiki['representation'] # 1550 x 1990
atoms = enwiki['atoms'][0] # 477
WordVector = enwiki['WordVector'] # 300 x 1990

#do random dimension reduction on all the things 
#300 -> 100, 50, 20

eps = 0.01

# after dimension reduction, wordVector becomes irrelevant: Only Dictionary and representation;
# Dictionary * representation = new_wordVector after dimension reduction

# dimension reduce Dictionary

#Calculate original distance matrix for cosine distances on the 477 atoms we care about
orig_dists = pairwise_distances(Dictionary.T, metric='cosine')
print orig_dists.shape
orig_477_dists = orig_dists[atoms, :]
orig_477_dists = orig_477_dists[:, atoms]
print np.shape(orig_477_dists)
num_dists = float(len(np.ravel(orig_477_dists)))
print "num_dists: " + str(num_dists)

# 100 dims
print("Using 100 dims...\n===================\n")
l1_err_in_cosdist = 1
while(l1_err_in_cosdist > 0.0641): # empirically found to be close to the lower bound
	transformer100 = GaussianRandomProjection(n_components=100, eps=eps)
	D_100 = transformer100.fit_transform(Dictionary.T)
	D_100 = D_100.T
	# calculate new dists
	dists_100dim = pairwise_distances(D_100.T, metric='cosine')
	dists_477_100dim = dists_100dim[atoms, :]
	dists_477_100dim = dists_477_100dim[:, atoms]
	l1_err_in_cosdist = norm(np.ravel(orig_477_dists) - np.ravel(dists_477_100dim), 1)/num_dists
	print "l1 error in cosine dist for 477 we care about for 100dims: " + str(l1_err_in_cosdist)
print("Final l1 error in cosine dist for 477 atoms for 100dims: " + str(l1_err_in_cosdist))

print("Using 50 dims...\n===================\n")
l1_err_in_cosdist = 1
while(l1_err_in_cosdist > 0.092): #empirically found to be close to the lower bound
	transformer50 = GaussianRandomProjection(n_components=50, eps=eps)
	D_50 = transformer50.fit_transform(Dictionary.T)
	D_50 = D_50.T
	# calculate new dists
	dists_50dim = pairwise_distances(D_50.T, metric='cosine')
	dists_477_50dim = dists_50dim[atoms, :]
	dists_477_50dim = dists_477_50dim[:, atoms]
	l1_err_in_cosdist = norm(np.ravel(orig_477_dists) - np.ravel(dists_477_50dim), 1)/num_dists
	print "l1 error in cosine dist for 477 we care about for 50dims: " + str(l1_err_in_cosdist)
print("Final l1 error in cosine dist for 477 atoms for 50dims: " + str(l1_err_in_cosdist))

print("Using 20 dims...\n===================\n")
l1_err_in_cosdist = 1
while(l1_err_in_cosdist > 0.144): # empirically found to be close to the lower bound
	transformer20 = GaussianRandomProjection(n_components=20, eps=eps)
	D_20 = transformer20.fit_transform(Dictionary.T)
	D_20 = D_20.T
	# calculate new dists
	dists_20dim = pairwise_distances(D_20.T, metric='cosine')
	dists_477_20dim = dists_20dim[atoms, :]
	dists_477_20dim = dists_477_20dim[:, atoms]
	l1_err_in_cosdist = norm(np.ravel(orig_477_dists) - np.ravel(dists_477_20dim), 1)/num_dists
	print "l1 error in cosine dist for 477 we care about for 20dims: " + str(l1_err_in_cosdist)
print("Final l1 error in cosine dist for 477 atoms for 20dims: " + str(l1_err_in_cosdist))

# now re-build word vectors
print("Generating low-dimensional word vectors...")

wv_100 = np.dot(D_100, representation)
wv_50 = np.dot(D_50, representation)
wv_20 = np.dot(D_20, representation)

print("Saving...")


np.savez('jl_dim_enwiki_subtract1_3sparse_477atoms_vecs.npz', words=words, corr=corr, corr_words=corr_words, representation= representation, atoms=atoms, D_100=D_100, D_50=D_50, D_20=D_20, D_300=Dictionary, wv_100=wv_100, wv_50 = wv_50, wv_20 = wv_20, wv_300 = WordVector)












