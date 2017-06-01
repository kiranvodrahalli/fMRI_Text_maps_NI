import numpy as np
import pickle as p
from numpy.linalg import svd, norm
import os
#import pylab as pl
from scipy import stats
from scipy.stats import pearsonr, sem
from scipy.io import loadmat
from random import shuffle
from path_to_RESULTS import PATH_TO_RESULTS

#main_path = '/jukebox/norman/knv/thesis/sherlock/RESULTS/'
main_path = PATH_TO_RESULTS + 'RESULTS/'
# possible to-do: implement a convolution to see if it improves performance
#                 as in mitchell on hp-dataset

def add_prev_time_steps(data, num_time_steps_in_past):
    timesteps = data.shape[1] # assume that data is features x timesteps
    orig_num_features = data.shape[0]
    added_mat = None
    for t in range(timesteps):
        # let's put the most recent one on the top, and go backwards from there
        # (building a column: [curr, curr - 1, curr - 2, curr - 3, ..., curr - k]^T)
        curr_rep = np.reshape(data[:, t].copy(), (orig_num_features, 1))
        for k in range(1, num_time_steps_in_past+1):
            curr_step_past = np.zeros((orig_num_features, 1))
            if t -k >= 0:
                curr_step_past = np.reshape(data[:, t - k].copy(), (orig_num_features, 1))
            curr_rep = np.r_[curr_rep, curr_step_past]
        assert(curr_rep.shape == ((num_time_steps_in_past + 1)*orig_num_features, 1))
        if added_mat is None:
            added_mat = curr_rep
        else:
            added_mat = np.c_[added_mat, curr_rep]
    assert(added_mat.shape == ((num_time_steps_in_past + 1)*orig_num_features, timesteps))
    return added_mat

def keep_only_first_time_step(data, num_time_steps_in_past):
    timesteps = data.shape[1]
    added_num_features = data.shape[0]
    assert(added_num_features % (num_time_steps_in_past + 1) == 0)
    orig_num_features = added_num_features/ (num_time_steps_in_past + 1)
    orig_data = data[0:orig_num_features, :]
    assert(orig_data.shape == (orig_num_features, timesteps))
    return orig_data

""" Compute projections on the positive simplex or the L1-ball

A positive simplex is a set X = { \mathbf{x} | \sum_i x_i = s, x_i \geq 0 }

The (unit) L1-ball is the set X = { \mathbf{x} | || x ||_1 \leq 1 }

Adrien Gaidon - INRIA - 2011
"""
# assume sum = 1 for simplex
def euclidean_proj_simplex(v, s=1):
    """ Compute the Euclidean projection on a positive simplex

    Solves the optimisation problem (using the algorithm from [1]):

        min_w 0.5 * || w - v ||_2^2 , s.t. \sum_i w_i = s, w_i >= 0 

    Parameters
    ----------
    v: (n,) numpy array,
       n-dimensional vector to project

    s: int, optional, default: 1,
       radius of the simplex

    Returns
    -------
    w: (n,) numpy array,
       Euclidean projection of v on the simplex

    Notes
    -----
    The complexity of this algorithm is in O(n log(n)) as it involves sorting v.
    Better alternatives exist for high-dimensional sparse vectors (cf. [1])
    However, this implementation still easily scales to millions of dimensions.

    References
    ----------
    [1] Efficient Projections onto the .1-Ball for Learning in High Dimensions
        John Duchi, Shai Shalev-Shwartz, Yoram Singer, and Tushar Chandra.
        International Conference on Machine Learning (ICML 2008)
        http://www.cs.berkeley.edu/~jduchi/projects/DuchiSiShCh08.pdf
    """
    assert s > 0, "Radius s must be strictly positive (%d <= 0)" % s
    n, = v.shape  # will raise ValueError if v is not 1-D
    # check if we are already on the simplex
    if v.sum() == s and np.alltrue(v >= 0):
        # best projection: itself!
        return v
    # get the array of cumulative sums of a sorted (decreasing) copy of v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    # get the number of > 0 components of the optimal solution
    rho = np.nonzero(u * np.arange(1, n+1) > (cssv - s))[0][-1]
    # compute the Lagrange multiplier associated to the simplex constraint
    theta = (cssv[rho] - s) / (rho + 1.0)
    # compute the projection by thresholding v using theta
    w = (v - theta).clip(min=0)
    return w


def create_k_diag_conv_mat_from_weights(init, weights, num_times):
    C = init
    convolution_weights_len = weights.shape[0] # assume num_vars x 1
    for t in range(num_times):
        for j in range(t, min(t + convolution_weights_len, num_times)):
            C[t, j] = weights[j - t]
    return C



# solving X = WYC for W, C
# ONLY USE PROCRUSTES; it's better anyways
# where X = vox x time fMRI matrix
# where Y = featurex x time semantic embedding matrix
# W = linear map (ridge or procrustes), size (vox x features)
# C = time weighting matrix for word vectors (want to combine word vectors over time at different weights)
# C is size (time x time)
# for C, we constrain it to be the same weight for the position in the previous timestep
# for instance, if there are 4 total times in our representation, we learn 4 separate weights in C
# there are only 4 variables in C.
# we solve this with alternating minimization over W and C; solving for W exactly for each C
# and updating C with gradient descent.
# only do this for text -> fMRI
# eta: learning weight for gradient descent portion (learning C)
# direction: text -> fMRI "tf": X = WYC
# direction: fMRI -> text "ft": WX = YC ("Y = X, X = YC")
def learn_matrix_factorization_for_time_text_weights(train_times, test_times, fMRI_data, fMRI_param_str, semantic_data, semantic_param_str, eta=0.2, num_grad_its=50, num_AM_its=30, num_prev_times=10):
    convolution_weights_len = num_prev_times + 1
    num_times = len(train_times)
    # otherwise this doesn't make sense
    assert(convolution_weights_len < num_times)
    # initialize C with uniform weights
    init_weight = 1.0/convolution_weights_len
    weights = np.zeros(convolution_weights_len)
    for i in range(0, convolution_weights_len):
        weights[i] = init_weight
    C = create_k_diag_conv_mat_from_weights(np.zeros((num_times, num_times)), weights, num_times)
    print "init weights = " + str(weights)


    # which regression? only use procrustes 
    def regress(tr_data):
        return procrustes_fit(tr_data)

    # text -> fMRI
    #tr_tf = [fMRI_data[:, train_times], semantic_data[:, train_times]]
    #tst_tf = [fMRI_data[:, test_times], semantic_data[:, test_times]]

    X = fMRI_data[:, train_times]
    num_vox = fMRI_data.shape[0]
    Y = semantic_data[:, train_times]
    num_feats = semantic_data.shape[0]

    W_ft = None
    W_tf = None
    # begin AM-loop (alternating minimization)
    # learn text -> fMRI first 
    # more efficient to learn in reduce fMRI space (only 20 features)
    for it in range(num_AM_its): # 50
        print "AM_it# " + str(it)
        YC = np.dot(Y, C)
        #print "YC.shape = " + str(YC.shape)
        # update W
        W = regress([X, YC]) # returns a list of one matrix
        W = W[0]
        WY = np.dot(W, Y)
        
        #print "WY.shape = " + str(WY.shape)
        assert(len(WY.shape) == 2)

        # remove the weird shape issue
        # WY = WY[0, :, :]
        
        # update C (update the weights and thus C)
        for grad_it in range(num_grad_its): # 100
            print "grad_it# " + str(grad_it)
            # change the loss mat into the loss derivative mat (part of chain rule)
            # then variable specific parts in a loop
            loss_mat = 2.*(X - np.dot(WY, C))
            print "curr loss = " + str(np.sum((1./2.)*loss_mat))
            #print "Loss mat shape = " + str(loss_mat.shape)
            # doing full gradient descent by looping over all time steps per iteration
            # could make this stochastic to be more efficient (sample a random time for each voxel)
            # gradient for tied weights is just a sum over all gradient appearances (i.e. in convolution)
            for v in range(0, num_vox): # 20
                #print "voxel v = " + str(v)
                # sample minibatch of size 10 timepoints (data points)
                minibatch = np.random.choice(range(0, num_times), 10)
                for t in minibatch: #10 #range(0, num_times): #986
                    #print "time point t = " + str(t)
                    for k in range(0, convolution_weights_len): # 30
                        #print "k = " + str(k)
                        if t - k >= 0:
                            # the gradient for the weight at c_index
                            grad = loss_mat[v, t]*WY[v, t - k]
                            #k is the weights_index
                            weights[k] -= eta*grad
                        # otherwise, no update
            # project onto simplex
            weights = euclidean_proj_simplex(weights)
            # update C
            C = create_k_diag_conv_mat_from_weights(C, weights, num_times)
            print "weights = " + str(weights)
    YC = np.dot(Y, C)
    #print "final YC shape = " + str(YC.shape)
    # update W
    W = regress([X, YC]) # returns a list of one matrix
    W = W[0]
    W_tf = W # true in either case
    # now let's find W_ft
    # in this case, we can get the other direction just by taking transpose
    # since it's orthogonal
    W_ft = W.T
    return W_tf, W_ft, weights




# train_times: list of the time points which we train on 
# test_times: list of time points we test on
# learn fMRI -> text with orthogonal map
# learn text -> fMRI with ridge map
# need to make sure that fMRI and semantic data are shifted appropriately from each other before passing it in
def learn_linear_maps(train_times, test_times, fMRI_data, fMRI_param_str, semantic_data, semantic_param_str):
    # learn fMRI -> text (Y -> X)
    tr_ft = [semantic_data[:, train_times], fMRI_data[:, train_times]]
    tst_ft = [semantic_data[:, test_times], fMRI_data[:, test_times]]
    # ridge
    Wridge_ft = ridge_fit(1.0, tr_ft)
    # procrustes
    Wpro_ft = procrustes_fit(tr_ft)
    #---------------
    # learn text -> fMRI (Y -> X)
    tr_tf = [fMRI_data[:, train_times], semantic_data[:, train_times]]
    tst_tf = [fMRI_data[:, test_times], semantic_data[:, test_times]]
    # ridge
    Wridge_tf = ridge_fit(1.0, tr_tf)
    # procrustes
    Wpro_tf = procrustes_fit(tr_tf)

    # saving
    outpath = main_path + 'EXP/'+'fMRI_' + fMRI_param_str + '/' + 'Sem_' + semantic_param_str + '/'
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    np.savez(outpath + 'data+maps.npz', r_ft=Wridge_ft, r_tf=Wridge_tf, p_ft=Wpro_ft, p_tf=Wpro_tf, fmri_tr=tr_tf[0], fmri_tst=tst_tf[0], sem_tr=tr_ft[0], sem_tst=tst_ft[0])
    return Wridge_ft, Wridge_tf, Wpro_ft, Wpro_tf



'''
# solve for alpha vector:
# argmin_alpha sum_i = 1 ^n \| X_i\alpha - y_i\|_2^2
# X_i = WA_i, where W is map from fMRI -> text, A_i is the time chunk with previous time steps in text space
# and y_i is the fMRI at time i
# in other words, we're learning linear combinations of previous time steps in the text space (changing the text rep)
# to better fit the fMRI to it. 
# assume X_i is features x #prev time steps 
# assume y_i is features
# features are supposed to be in fMRI space
def learn_linear_combinations_of_vecs(X_list, y_list):
    Z = None
    D = None
    for i in range(len(X_list)):
        X = X_list[i]
        y = y_list[i]
        xlen = X.shape[1]
        if i == 0:
            Z = np.dot(y, X)
            D = np.zeros(xlen)
        else:
            Z += np.dot(y, X)
        for j in range(xlen):
            Xcolj = X[:, j]
            D[j] += np.linalg.norm(Xcolj) * np.linalg.norm(Xcolj)
    alpha = np.zeros(xlen)
    for j in range(xlen):
        alpha[j] = Z[j]/float(D[j])
    return alpha
'''    



# predict voxels from words X = WY
# where X = voxels x TRs, Y = features x TRs, W = voxels x features 

# We learn the map Y -> X, where Y is the last element and X is the first element

# (this is a setting where there are only two elements in data_list)
# to reverse the order, just flip X and Y to learn X -> Y
def ridge_fit(ridge_param, data_list):
    assert(len(data_list) == 2)
    # context vectors
    vec = data_list[len(data_list) - 1] # features x TRs
    YT = vec.T #TRs x features
    #print "YT.shape = " + str(YT.shape)
    num_features = np.shape(YT)[1]
    num_TRs = np.shape(YT)[0]
    #print "num features: " + str(num_features)
    #print "num_TRs: " + str(num_TRs)
    # YT = U \Sigma V^T
    U, s, VT = svd(YT, full_matrices=False)
    V = VT.T
    scaled_d = np.zeros(np.shape(s))
    for i in range(0, len(scaled_d)):
        scaled_d[i] = s[i]/(s[i]*s[i] + ridge_param)
    #print "scaled_d.shape = " + str(scaled_d.shape)
    #print "U.shape = " + str(U.shape)
    #print "V.shape = " + str(V.shape)
    # list of transforms, W is list of Wi which are W = voxels x features
    W = []
    for i in range(0, len(data_list) - 1):
        Xi = data_list[i].T #TRs x voxels
        #print "Xi.shape = " + str(Xi.shape)
        num_voxels_i = np.shape(Xi)[1]
        Wi = np.zeros((num_features, num_voxels_i))
        for k in range(0, num_voxels_i):
            xi = Xi[:, k]
            UTxi = np.dot(U.T, xi)
            #print "UTxi.shape = " + str(UTxi.shape)
            UTxi = np.ravel(UTxi) # NECESSARY TO HANDLE WEIRD MATRIX TYPES
            #print "UTxi.shape = " + str(UTxi.shape)
            dUTxi = scaled_d*UTxi
            #print "scaled_d*UTxi.shape = " + str(dUTxi.shape)
            w_k = np.dot(V, dUTxi)
            Wi[:, k] = w_k
        W.append(Wi.T)
        #print("Matrix shape: " + str(np.shape(Wi.T)))
    return W


# predict voxels from words X = WY
# where X = voxels x TRs, Y = features x TRs, W = voxels x features 
# We learn the map Y -> X, where Y is the last element and X is the first element
# (this is a setting where there are only two elements in data_list)
# since this is orthogonal, the reverse map is given by WT
def procrustes_fit(data_list):
    assert(len(data_list) == 2)
    # context vectors
    vec = data_list[len(data_list) - 1] # features x TRs
    YT = vec.T #TRs x features
    num_features = np.shape(YT)[1]
    num_TRs = np.shape(YT)[0]
    #print "num features: " + str(num_features)
    #print "num_TRs: " + str(num_TRs)

    W = []
    for i in range(0, len(data_list) - 1):
        Xi = data_list[i] #voxels x TRs
        num_voxels_i = np.shape(Xi)[0]
        #print "Num voxels: " + str(num_voxels_i)
        M = np.dot(Xi, YT)
        U, s, VT = svd(M, full_matrices=False)
        Wi = np.dot(U, VT) # num_voxels x num_features
        W.append(Wi)
        #print("Matrix shape: " + str(np.shape(Wi)))
    return W