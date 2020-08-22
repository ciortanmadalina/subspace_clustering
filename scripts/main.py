
import sys

import pandas as pd
import scripts.feature_ranking as feature_ranking
import scripts.feature_sampling as feature_sampling
import scripts.features_2d as features_2d
import scripts.ga as ga
import scripts.ga_evaluation as ga_evaluation
import scripts.mutations as mutations
import scripts.semi_supervised as semi_supervised
from tqdm import tqdm

sys.path.append("..")

def discovery_analysis(data,
                       truth,
                       n_clusters,
                       nb_top_subspaces=10,
                       sampling=None,
                       allow_subspace_overlap=True,
                       redundant_threshold=0.5,
                       round_size=4,
                       metric='correlation',
                       method="adapted_ratkowsky_lance",
                       clustering="hdbscan",
                       max_ranking_2d=None):
    """Performs the unsupervised analysis and returns top
    nb_top_subspaces identified subspaced. 

    Arguments:
        data {[type]} -- [description]
        truth {[type]} -- [description]
        n_clusters {[type]} -- [description]

    Keyword Arguments:
        nb_top_subspaces {int} -- [description] (default: {10})
        sampling {[type]} -- [description] (default: {None})
        allow_subspace_overlap {bool} -- [description] (default: {True})
        redundant_threshold {float} -- [description] (default: {0.5})
        round_size {int} -- [description] (default: {4})
        metric {str} -- [description] (default: {'correlation'})
        method {str} -- [description] (default: {"adapted_ratkowsky_lance"})
        clustering {str} -- [description] (default: {"hdbscan"})
        max_ranking_2d {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    print("*** Unsupervised Analysis")
    meta_features = feature_ranking.rank_features(
        data,
        nb_bins=20,
        rank_threshold=85,
        z_file=None,
        metric=metric,
        redundant_threshold=redundant_threshold)

    #     population, n = features_2d.run(data,
    #                                     None if clustering == "hdbscan" else n_clusters,
    #                                     meta_features,
    #                                     model_file=model_file,
    #                                     add_close_population=False)
    # GMM + Adapted ratkowski lance performs better than hdbscan, even when the number
    # of clusters is not determined
    model_file_2d = 'models/gmm_arl.h5'
    population, _ = features_2d.run(data,
                                    n_clusters,
                                    meta_features,
                                    model_file=model_file_2d,
                                    add_close_population=False)
    if max_ranking_2d is None:
        max_ranking_2d = data.shape[1] // 4

    if sampling is None:
        sampling = {
            "ARCHIVE2D": {
                "ga": 0.3,
                "max": 0.3
            },
            "CLOSE": {
                "ga": 0.4,
                "max": 0.4
            },
            "IMP1D": {
                "ga": 0.2,
                "max": 0.2
            },
            "RANDOM": {
                "ga": 0.1,
                "max": 0.1
            },
        }
    params = ga.ga_parameters(n_clusters,
                              data.shape[1],
                              truth,
                              meta_features,
                              method=method,
                              truth_methods=['ari'],
                              archive_2d=population.iloc[:max_ranking_2d],
                              epochs=nb_top_subspaces * round_size,
                              round_size=round_size,
                              allow_subspace_overlap=allow_subspace_overlap,
                              improvement_per_mutation_report=False,
                              sampling=sampling,
                              clustering=clustering)
    solutions, _ = ga.run(data, params)
    return solutions

def semi_supervised_analysis(seeds,
                             data,
                             truth,
                             n_clusters,
                             sampling=None,
                             debug=False,
                             allow_subspace_overlap=True,
                             redundant_threshold=0.5,
                             metric='correlation',
                             method="adapted_ratkowsky_lance",
                             model_file=f'models/gmm_arl.h5',
                             clustering="gmm"):
    """Performs the semi-supervised analysis for each of the
    starting subsets in seeds.

    Arguments:
        seeds {[type]} -- [description]
        data {[type]} -- [description]
        truth {[type]} -- [description]
        n_clusters {[type]} -- [description]

    Keyword Arguments:
        sampling {[type]} -- [description] (default: {None})
        debug {bool} -- [description] (default: {False})
        allow_subspace_overlap {bool} -- [description] (default: {True})
        redundant_threshold {float} -- [description] (default: {0.5})
        metric {str} -- [description] (default: {'correlation'})
        method {str} -- [description] (default: {"adapted_ratkowsky_lance"})
        model_file {[type]} -- [description] (default: {f'models/gmm_arl.h5'})
        clustering {str} -- [description] (default: {"gmm"})

    Returns:
        [type] -- [description]
    """
    print("*** Semi Supervised Analysis")
    meta_features = feature_ranking.rank_features(
        data,
        nb_bins=20,
        rank_threshold=85,
        z_file=None,
        metric=metric,
        redundant_threshold=redundant_threshold)

    #     population, n = features_2d.run(data,
    #                                     None if clustering == "hdbscan" else n_clusters,
    #                                     meta_features,
    #                                     model_file=model_file,
    #                                     add_close_population=False)
    # GMM + Adapted ratkowski lance performs better than hdbscan, even when the number
    # of clusters is not determined
    model_file_2d = 'models/gmm_arl.h5'
    population, n = features_2d.run(data,
                                    n_clusters,
                                    meta_features,
                                    model_file=model_file_2d,
                                    add_close_population=False)

    solutions = pd.DataFrame()
    for seed in seeds:
        best_subspace = semi_supervised.maximize_subspace(
            seed,
            data,
            truth,
            n_clusters,
            meta_features,
            population[["feature1", "feature2"]],
            sampling=sampling,
            debug=debug,
            allow_subspace_overlap=allow_subspace_overlap,
            clustering=clustering,
            method=method)
        print(f"Found {best_subspace}")
        solutions = solutions.append(best_subspace,
                                     sort=False,
                                     ignore_index=True)
    return solutions

def evaluate_subspace(subspace,
                      data,
                      clustering,
                      n_clusters,
                      truth,
                      methods=["adapted_ratkowsky_lance", "silhouette"]):
    params = {
        "n_clusters": n_clusters,
        "min_cluster_size": 4,
        "methods": ["adapted_ratkowsky_lance", "silhouette"],
        "truth_methods": ["ari"],
        "y": truth,
        "clustering": clustering
    }


    return ga_evaluation.clustering_evaluation(subspace, data, params)
