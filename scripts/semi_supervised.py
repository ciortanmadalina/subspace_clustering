
import sys

import numpy as np
import pandas as pd
import scripts.feature_sampling as feature_sampling
import scripts.ga as ga
import scripts.ga_evaluation as ga_evaluation
import scripts.mutations as mutations

sys.path.append("..")


def maximize_subspace( seed, data, truth, n_clusters, meta_features,archive, 
                      sampling = None, debug = False,
                      allow_subspace_overlap = True, 
                      clustering = "hdbscan", method = "adapted_silhouette"):

    if sampling is None:
        sampling = {
            "ARCHIVE2D": { 
                "ga": 0.3,
                "max": 0.3 },
            "CLOSE": { 
                "ga": 0.4,
                "max": 0.4 },
            "IMP1D": { 
                "ga": 0.2,
                "max": 0.2 },
            "RANDOM": { 
                "ga": 0.1,
                "max": 0.1},
        }
    params = ga.ga_parameters(
        n_clusters,
        data.shape[1],
        truth,
        meta_features,
        truth_methods=['ari'],
        archive_2d=archive,
        allow_subspace_overlap = allow_subspace_overlap,
        improvement_per_mutation_report = False,
        sampling = sampling,
        debug = debug,
        clustering = clustering,
        method = method
        
    )
    indiv_scores, archive = ga_evaluation.evaluateIndividual(
                seed, data, params, archive)
    print(f"Initial subspace score {indiv_scores}")
    individual_str = "_".join(np.array(seed).astype(str))
    exploration = np.zeros(data.shape[1])
    best_individual, archive = ga.maximize_individual(
                individual_str, data, params, archive, exploration)
    best_individual = ga.prettify_best_individual(best_individual, data, params)
    return best_individual
