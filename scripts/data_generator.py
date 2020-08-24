import sys

import matplotlib.pyplot as plt
import numpy as np
import scripts.internal_scores as validation
import scripts.plot_utils as plot_utils
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import (make_blobs, make_circles,
                                                make_moons)
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score

sys.path.append("..")


def make_data_for_loss(n_subspaces=1,
                       n_clusters=5,
                       n_samples=None,
                       cluster_std=0.01,
                       random_state=0,
                       n_definingfeatures=4,
                       n_uniform_features=40,
                       n_normal_features=40,
                       plotPCA=True,
                       plotSilhouette=True):
    """[summary]

    Keyword Arguments:
        n_subspaces {int} -- [description] (default: {1})
        n_clusters {int} -- [description] (default: {5})
        n_samples {[type]} -- [description] (default: {None})
        cluster_std {float} -- [description] (default: {0.01})
        random_state {int} -- [description] (default: {0})
        n_definingfeatures {int} -- [description] (default: {4})
        n_uniform_features {int} -- [description] (default: {40})
        n_normal_features {int} -- [description] (default: {40})
        plotPCA {bool} -- [description] (default: {True})
        plotSilhouette {bool} -- [description] (default: {True})

    Returns:
        [type] -- [description]
    """
    truths = []
    if n_samples is None:
        n_samples = np.random.randint(30, 50) * n_clusters

    data = None
    for r in range(n_subspaces):
        data_x, truth = make_blobs(n_samples=n_samples,
                                   centers=n_clusters,
                                   random_state=random_state + r,
                                   n_features=n_definingfeatures,
                                   cluster_std=cluster_std,
                                   center_box=(0, 1))
        print(f"cluster_std {cluster_std}")
        truths.append(truth)
        pca = PCA(2)
        pca_data = pca.fit_transform(data_x)
        if plotPCA or plotSilhouette:
            plt.figure(figsize=(4 * (plotPCA + plotSilhouette), 3))
        if plotPCA:
            plt.subplot(1, plotPCA + plotSilhouette, 1)
            plt.title("PCA GA important features")
            plt.scatter(pca_data[:, 0], pca_data[:, 1], c=truth)
        if plotSilhouette:
            ax = plt.subplot(1, plotPCA + plotSilhouette,
                             plotPCA + plotSilhouette)
            plot_utils.silhouette_plot(pca_data, truths[r], ax=ax)

        predK = KMeans(n_clusters=n_clusters).fit(data_x).labels_
        ari = adjusted_rand_score(truth, predK)

        print(f'ARI  {round(ari,2)}')
        if data is None:
            data = data_x
        else:
            data = np.hstack([data, data_x])

    uniform_features = np.random.uniform(low=0.1,
                                         high=0.9,
                                         size=(
                                             n_samples,
                                             n_uniform_features,
                                         ))
    normal_features = np.random.normal(size=(
        n_samples,
        n_normal_features,
    ))

    data = np.hstack([data, uniform_features, normal_features])
    data = preprocessing.MinMaxScaler().fit_transform(data)

    return data, truths


def optimal_number_of_clusters(data, nMax=8, random_state=0):
    """[summary]

    Arguments:
        data {[type]} -- [description]

    Keyword Arguments:
        nMax {int} -- [description] (default: {8})
        random_state {int} -- [description] (default: {0})

    Returns:
        [type] -- [description]
    """
    wcss = []
    for n in range(2, nMax):
        kmeans = KMeans(n_clusters=n, random_state=random_state)
        kmeans.fit_predict(X=data)
        wcss.append(kmeans.inertia_)
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss) - 1]

    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
        distances.append(numerator / denominator)
    n = distances.index(max(distances)) + 2
    return n


def make_multivariate_gaussian(n_centers, n_features, n_samples, cov_max = -1, random_state = 0):
    """
    When cov_max = -1 => isotropic blobs
    """
    from sklearn.utils import check_random_state
    
    generator = check_random_state(random_state)
    center_box = (-10, 10)
    centers = generator.uniform(center_box[0], center_box[1],
                                        size=(n_centers, n_features))
    if cov_max == -1:
        d_covs = [np.eye(n_features) for _ in range(n_centers)]
    else:
        d_covs = []
        cov = 0.1 + abs(cov_max)/10
        print(f"COV {cov} ")
        for ii in range(n_centers):
            
            v = np.random.uniform(0, cov, 
                    size = (n_features, n_features))
            v = np.dot(v,v.transpose())
            np.fill_diagonal(v, 0)
            v += np.eye(n_features) * np.random.uniform(np.max(v), abs(cov_max),
                                                      size = n_features)
            
            d_covs.append(v)
#     print(d_covs)
    data = []
    truth = []
    ns = n_samples// n_centers
    ns = [ns] * ( n_centers-1)
    ns.append(n_samples - np.sum(ns))
    for cluster in range(len(centers)):
        np.random.seed(cluster)
        arr = np.random.multivariate_normal(centers[cluster], d_covs[cluster], ns[cluster])
        data.extend(arr)
        truth.extend([cluster] *  ns[cluster])
    data = np.array(data)
    truth = np.array(truth)
    return data, truth

def make_blob_data(n_samples,
                   n_clusters_per_subpace,
                   cluster_std=None,
                   min_subspace_features=4,
                   max_subspace_features=6,
                   plot=False,
                   isotropic = True):
    """Generate pairs of features consisting of Gaussian blob data

    Arguments:
        n_samples {[type]} -- [description]
        n_clusters_per_subpace {[type]} -- [description]

    Keyword Arguments:
        cluster_std {[type]} -- [description] (default: {None})
        min_subspace_features {int} -- [description] (default: {4})
        max_subspace_features {int} -- [description] (default: {6})
        plot {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    if cluster_std is None:
        cluster_std = np.random.uniform(0.01, 0.14,
                                        len(n_clusters_per_subpace))

    data = None

    best_subspaces = []  # best features
    truths = []  # ground truth
    nb_subspaces = len(n_clusters_per_subpace)
    for k, random_state in enumerate(range(nb_subspaces)):
        n_definingfeatures = np.random.randint(min_subspace_features,
                                               max_subspace_features)

        n_clusters = n_clusters_per_subpace[k]
        if isotropic == True:
            data_x, truth = make_blobs(n_samples=n_samples,
                                   centers=n_clusters,
                                   random_state=random_state,
                                   n_features=n_definingfeatures,
                                   cluster_std=cluster_std[k],
                                   center_box=(0, 1))
        else:
            data_x, truth= make_multivariate_gaussian(
                n_clusters, n_definingfeatures, n_samples, cov_max = cluster_std[k], random_state = 0)
            
        idx = np.arange(len(truth))
        np.random.shuffle(idx)
        data_x = data_x[idx]
        truth = truth[idx]

        truths.append(truth)
        if data is None:
            data = data_x
            best_subspaces.append(np.arange(data.shape[1]))
        else:
            data = np.hstack([data, data_x])
            best_subspaces.append(
                np.arange(data.shape[1] - data_x.shape[1], data.shape[1]))

        if plot:
            pca = PCA(2)
            pca_data = pca.fit_transform(data_x)
            plt.figure()
            plt.title(f"PCA {best_subspaces[-1]}")
            plt.scatter(pca_data[:, 0], pca_data[:, 1], c=truth)

            predK = KMeans(n_clusters=n_clusters).fit(data_x).labels_
            ari = adjusted_rand_score(truth, predK)

            plot_utils.silhouette_plot(data[:, best_subspaces[-1]], truth)
            print(f'ARI  {round(ari,2)}, std {cluster_std[k]}')
            plt.show()
    return data, best_subspaces, truths


def make_redundant_data(data, n_blob_data, n_samples, n_redundant,
                        random_redundant):
    """Introduce reundancy in the dataset by randomly selecting
    features to copy and alter by adding noise

    Arguments:
        data {[type]} -- [description]
        n_blob_data {[type]} -- [description]
        n_samples {[type]} -- [description]
        n_redundant {[type]} -- [description]
        random_redundant {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    redundant_data = np.array([])
    if n_redundant > 0:
        noise = np.random.uniform(low=0.01,
                                  high=np.random.uniform(low=0.1, high=0.6),
                                  size=(
                                      n_samples,
                                      n_redundant,
                                  ))
        if random_redundant:
            redundant_idx = np.random.choice(np.arange(data.shape[1]),
                                             replace=False,
                                             size=n_redundant)
        else:
            redundant_idx = np.random.choice(np.arange(n_blob_data),
                                             replace=False,
                                             size=n_redundant)
        redundant_data = data[:, redundant_idx] + noise
    return redundant_data


def make_negative_binomial(n_neg_binomial, n_samples, max_neg_bin_p):
    """Simulate negative binomial data.
    Arguments:
        n_neg_binomial {[type]} -- [description]
        n_samples {[type]} -- [description]
        max_neg_bin_p {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    n = np.random.uniform(0.5, 10, n_neg_binomial)
    p = np.random.uniform(0.05, max_neg_bin_p, n_neg_binomial)
    neg_binomial_features = np.random.negative_binomial(
        n, p, (n_samples, n_neg_binomial))
    return neg_binomial_features


def make_beta(n_beta, n_samples):
    """Simulate Beta distributions.

    Arguments:
        n_beta {[type]} -- [description]
        n_samples {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    alpha = np.random.uniform(0.1, 4, n_beta)
    beta = np.random.uniform(0.5, 5, n_beta)
    beta_features = np.random.beta(alpha, beta, (n_samples, n_beta))
    return beta_features


def make_gamma(n_gamma, n_samples):
    """Simulate Gamma Distributions

    Arguments:
        n_gamma {[type]} -- [description]
        n_samples {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    n = np.random.uniform(0.5, 5, n_gamma)
    p = np.random.uniform(0.01, 0.9, n_gamma)
    gamma_features = np.random.gamma(n, p, (n_samples, n_gamma))
    return gamma_features


def make_bimodal(n_bimodal_features, n_samples):
    """Simulate bimodal data

    Arguments:
        n_bimodal_features {[type]} -- [description]
        n_samples {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    bimodal_features = np.concatenate([
        np.random.negative_binomial(1, 0.15,
                                    (n_samples // 2, n_bimodal_features)),
        100 -
        np.random.negative_binomial(1, 0.15,
                                    (n_samples -
                                     (n_samples // 2), n_bimodal_features))
    ])
    return bimodal_features


def add_outliers_to_distribution(n_outlier_features, n_samples, dist):
    """Add outliers to arbitrary distributions (specified using the 
    dist parameter)

    Arguments:
        n_outlier_features {[type]} -- [description]
        n_samples {[type]} -- [description]
        dist {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if n_outlier_features == 0:
        return np.zeros((n_samples, 0))
    n_outliers = np.random.randint(1, 20)
    if dist == "uniform":
        source_features = np.random.uniform(low=0.1,
                                            high=0.9,
                                            size=(n_samples - n_outliers,
                                                  n_outlier_features))
    if dist == "normal":
        source_features = np.random.normal(size=(n_samples - n_outliers,
                                                 n_outlier_features))
    if dist == "negative_binomial":
        source_features = 100 - np.random.negative_binomial(
            1, 0.15, (n_samples - n_outliers, n_outlier_features))

    delta = np.max(source_features) - np.min(source_features)

    if np.random.randint(2) == 0:  # add to the right
        outlier_features = np.vstack([
            source_features,
            np.random.uniform(low=np.max(source_features) + delta * 0.2,
                              high=np.max(source_features) + delta,
                              size=(
                                  n_outliers,
                                  n_outlier_features,
                              ))
        ])
    else:
        outlier_features = np.vstack([
            source_features,
            np.random.uniform(low=np.min(source_features) - delta,
                              high=np.min(source_features) - delta * 0.2,
                              size=(
                                  n_outliers,
                                  n_outlier_features,
                              ))
        ])
    return outlier_features


def cutoff_data(data, n_cutoff):
    """Create RNA-seq specific patterns by clipping input data
    to an arbitrary value.

    Arguments:
        data {[type]} -- [description]
        n_cutoff {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if n_cutoff == 0:
        return data
    idx = np.random.choice(np.arange(data.shape[1]), n_cutoff)
    for i in idx:
        if np.random.choice(2) == 0:  #cut above
            threshold = np.percentile(data[:, i], np.random.randint(65, 95))
            data[np.where(data[:, i] > threshold)[0], i] = threshold
        else:
            threshold = np.percentile(data[:, i], np.random.randint(5, 45))
            data[np.where(data[:, i] < threshold)[0], i] = threshold
    return data


def make_data_for_ga(n_clusters_per_subpace,
                     cluster_std=None,
                     n_uniform_features=0,
                     n_normal_features=0,
                     n_neg_binomial=0,
                     max_neg_bin_p=0.9,
                     n_gamma=0,
                     n_beta=0,
                     random_redundant=True,
                     n_redundant=1,
                     n_outlier_features=0,
                     n_cutoff=0,
                     n_bimodal_features=0,
                     min_subspace_features=4,
                     max_subspace_features=6,
                     n_samples=None,
                     plot=False,
                     isotropic = True):
    """Generates a simulated dataset consisting of a mixture of
    distributions selected through input parameters

    Arguments:
        n_clusters_per_subpace {[type]} -- [description]

    Keyword Arguments:
        cluster_std {[type]} -- [description] (default: {None})
        n_uniform_features {int} -- [description] (default: {0})
        n_normal_features {int} -- [description] (default: {0})
        n_neg_binomial {int} -- [description] (default: {0})
        max_neg_bin_p {float} -- [description] (default: {0.9})
        n_gamma {int} -- [description] (default: {0})
        n_beta {int} -- [description] (default: {0})
        random_redundant {bool} -- [description] (default: {True})
        n_redundant {int} -- [description] (default: {5})
        n_outlier_features {int} -- [description] (default: {0})
        n_cutoff {int} -- [description] (default: {0})
        n_bimodal_features {int} -- [description] (default: {0})
        min_subspace_features {int} -- [description] (default: {4})
        max_subspace_features {int} -- [description] (default: {6})
        n_samples {[type]} -- [description] (default: {None})
        plot {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    if n_samples is None:
        n_samples = np.random.randint(10, 150) * max(n_clusters_per_subpace)

    data, best_subspaces, truths = make_blob_data(n_samples,
                                                  n_clusters_per_subpace,
                                                  cluster_std,
                                                  min_subspace_features,
                                                  max_subspace_features, plot, isotropic)
    n_blob_data = data.shape[1]

    neg_binomial_features = make_negative_binomial(n_neg_binomial, n_samples,
                                                   max_neg_bin_p)
    beta_features = make_beta(n_beta, n_samples)
    gamma_features = make_gamma(n_gamma, n_samples)
    moon_features, _ = make_moons(n_samples=n_samples, noise=0.1)
    circle_features, _ = make_circles(n_samples=n_samples, noise=0.1)
    uniform_features = np.random.uniform(low=0.1,
                                         high=0.9,
                                         size=(n_samples, n_uniform_features))
    normal_features = np.random.normal(size=(n_samples, n_normal_features))
    bimodal_features = make_bimodal(n_bimodal_features, n_samples)

    outlier_uniform = add_outliers_to_distribution(n_outlier_features,
                                                   n_samples, "uniform")
    outlier_normal = add_outliers_to_distribution(n_outlier_features,
                                                  n_samples, "normal")
    outlier_binomial = add_outliers_to_distribution(n_outlier_features,
                                                    n_samples,
                                                    "negative_binomial")

    data = np.hstack([
        data,
        uniform_features,
        normal_features,
        neg_binomial_features,
        gamma_features,  #moon_features, circle_features, 
        bimodal_features,
        outlier_uniform,
        outlier_normal,
        outlier_binomial,
        beta_features
    ])

    redundant_data = make_redundant_data(data, n_blob_data, n_samples,
                                         n_redundant, random_redundant)
    data = np.hstack([data, redundant_data])
    data = cutoff_data(data, n_cutoff)

    print(
        f"Total: {data.shape}, uniform {uniform_features.shape} , normal {normal_features.shape},"
        +
        f"neg bin {neg_binomial_features.shape}, gamma {gamma_features.shape}, moon {moon_features.shape}, "
        +
        f"circle {circle_features.shape},  bimodal {bimodal_features.shape}, redundat {redundant_data.shape}, "
        +
        f"outliers with uniform {outlier_uniform.shape}, outliers with normal {outlier_normal.shape} ,"
        +
        f"outlier binomial {outlier_binomial.shape}, beta features {beta_features.shape}, "
    )

    data = preprocessing.MinMaxScaler().fit_transform(data)

    return data, best_subspaces, truths
