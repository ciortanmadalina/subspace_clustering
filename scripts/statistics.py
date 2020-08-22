from collections import Counter

import numpy as np
import scipy
import numpy.matlib
from scipy.sparse import csc_matrix
from sklearn.metrics.pairwise import rbf_kernel, cosine_similarity
from numpy import linalg as LA
from sklearn.neighbors import kneighbors_graph 
from scipy.sparse.linalg import expm
from scipy.sparse import diags
from scipy.stats import hypergeom

def overlap(subspace1_size, subspace2_size, overlap_size, dataset_size):
    return hypergeom.sf(overlap_size, dataset_size, subspace1_size, subspace2_size-1)

def mean_abs_difference(data):
    """
    https://scihub.bban.top/https://doi.org/10.1016/j.patrec.2012.05.019
    
    https://www.ijcsmc.com/docs/papers/July2016/V5I7201670.pdf
    """
    mad = np.sum(np.abs(data - np.mean(data, axis=0)), axis=0) / data.shape[1]
    return mad


def dispersion(data):
    """
    https://scihub.bban.top/https://doi.org/10.1016/j.patrec.2012.05.019
    
    https://www.ijcsmc.com/docs/papers/July2016/V5I7201670.pdf
    """
    data = data + 1  #avoid 0 division
    arithmetic_mean = np.mean(data, axis=0)
    geometric_mean = np.power(np.prod(data, axis=0), 1 / data.shape[1])
    R = arithmetic_mean / geometric_mean
    return R


def mean_median(data):
    """
    https://scihub.bban.top/https://doi.org/10.1016/j.patrec.2012.05.019
    
    https://www.ijcsmc.com/docs/papers/July2016/V5I7201670.pdf
    """
    mm = np.abs(np.mean(data, axis=0) - np.median(data, axis=0))
    return mm


def amam(data):
    """
    https://scihub.bban.top/https://doi.org/10.1016/j.patrec.2012.05.019
    
    https://www.ijcsmc.com/docs/papers/July2016/V5I7201670.pdf
    """
    result = np.sum(np.exp(data),
                    axis=0) / (np.exp(np.mean(data, axis=0)) * data.shape[1])
    return result


def compute_entropy(data, nb_bins=20):
    """
    Computes the entropy of a 1D array

    Arguments:
        data {[type]} -- [description]

    Keyword Arguments:
        nb_bins {int} -- [description] (default: {20})

    Returns:
        [type] -- [description]
    """
    bins = np.linspace(min(data), max(data), nb_bins)
    y = np.digitize(data, bins)
    y = list(Counter(np.ravel(y)).values())
    return scipy.stats.entropy(y, base=2) / np.log2(nb_bins)


def spec_scores( x):
    similarity = rbf_kernel(x)
    adjacency = similarity
    degree_vector = np.sum(adjacency, 1)
    degree = np.diag(degree_vector)
    laplacian = degree - adjacency
    normaliser_vector = np.reciprocal(np.sqrt(degree_vector))
    normaliser = np.diag(normaliser_vector)

    normalised_laplacian = normaliser.dot(laplacian).dot(normaliser)

    weighted_features = np.matmul(normaliser, x)

    normalised_features = weighted_features / np.linalg.norm(weighted_features, axis=0)
    all_to_all = normalised_features.transpose().dot(normalised_laplacian).dot(normalised_features)
    scores = np.diag(all_to_all)
    return scores

def construct_W(X, neighbour_size = 5, t = 1):
    n_samples, n_features = np.shape(X)
    S=kneighbors_graph(X, neighbour_size+1, mode='distance',metric='euclidean') #sqecludian distance works only with mode=connectivity  results were absurd
    S = (-1*(S*S))/(2*t*t)
    S=S.tocsc()
    S=expm(S) # exponential
    S=S.tocsr()

    #[1]  M. Belkin and P. Niyogi, “Laplacian Eigenmaps and Spectral Techniques for Embedding and Clustering,” Advances in Neural Information Processing Systems,
    #Vol. 14, 2001. Following the paper to make the weights matrix symmetrix we use this method

    bigger = np.transpose(S) > S
    S = S - S.multiply(bigger) + np.transpose(S).multiply(bigger)
    return S

def fisher_score(X, y):
    """
    This function implements the fisher score feature selection, steps are as follows:
    1. Construct the affinity matrix W in fisher score way
    2. For the r-th feature, we define fr = X(:,r), D = diag(W*ones), ones = [1,...,1]', L = D - W
    3. Let fr_hat = fr - (fr'*D*ones)*ones/(ones'*D*ones)
    4. Fisher score for the r-th feature is score = (fr_hat'*D*fr_hat)/(fr_hat'*L*fr_hat)-1
    Input
    -----
    X: {numpy array}, shape (n_samples, n_features)
        input data
    y: {numpy array}, shape (n_samples,)
        input class labels
    Output
    ------
    score: {numpy array}, shape (n_features,)
        fisher score for each feature
    Reference
    ---------
    He, Xiaofei et al. "Laplacian Score for Feature Selection." NIPS 2005.
    Duda, Richard et al. "Pattern classification." John Wiley & Sons, 2012.
    """

    # Construct weight matrix W in a fisherScore way
    kwargs = {"neighbor_mode": "supervised", "fisher_score": True, 'y': y}
#     W = construct_W(X, **kwargs)
    W = construct_W(X)

    # build the diagonal D matrix from affinity matrix W
    D = np.array(W.sum(axis=1))
    L = W
    tmp = np.dot(np.transpose(D), X)
    D = diags(np.transpose(D), [0])
    Xt = np.transpose(X)
    t1 = np.transpose(np.dot(Xt, D.todense()))
    t2 = np.transpose(np.dot(Xt, L.todense()))
    # compute the numerator of Lr
    D_prime = np.sum(np.multiply(t1, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # compute the denominator of Lr
    L_prime = np.sum(np.multiply(t2, X), 0) - np.multiply(tmp, tmp)/D.sum()
    # avoid the denominator of Lr to be 0
    D_prime[D_prime < 1e-12] = 10000
    lap_score = 1 - np.array(np.multiply(L_prime, 1/D_prime))[0, :]

    # compute fisher score from laplacian score, where fisher_score = 1/lap_score - 1
    score = 1.0/lap_score - 1
    score = np.transpose(score)
    idx = np.argsort(score, 0)
    return idx[::-1]
    
    