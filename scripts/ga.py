
import sys
from collections import Counter

import numpy as np
import pandas as pd
import scripts.feature_sampling as feature_sampling
import scripts.ga_evaluation as ga_evaluation
import scripts.mutations as mutations
from tqdm import tqdm

sys.path.append("..")

def ga_parameters(
    n_clusters,
    nb_dims,
    truth,
    meta_features,
    method="adapted_ratkowsky_lance",
    truth_methods=['ari'],  # maxAriPerClass
    archive_2d=None,
    epochs=40,
    round_size=4,
    debug=False,
    elitism=True,
    sampling=None,
    ignore_redundant=True,
    total_maximisation_exploration=300,
    allow_subspace_overlap=False,
    score_tolerance = 0.008,
    improvement_per_mutation_report = False,
    clustering = "hdbscan",
    multi_n_clusters = None,
    population_size = 50,
    max_subspace_size = 150,
    min_cluster_size = 4,
    maximisation_size = 100,
    pca = False,
    hdbscan_min_cluster_size = 10):
    """
    Set all parameters for the genetic algorithm

    Arguments:
        n_clusters {[type]} -- [description]
        nb_dims {[type]} -- [description]
        truth {[type]} -- [description]
        meta_features {[type]} -- [description]

    Keyword Arguments:
        method {str} -- [description] (default: {"adapted_ratkowsky_lance"})
        truth_methods {list} -- [description] (default: {['ari']})
        epochs {int} -- [description] (default: {3})
        round_size {int} -- [description] (default: {2})
        debug {bool} -- [description] (default: {True})
        elitism {bool} -- [description] (default: {True})
        sampling {[type]} -- [description] (default: {None})
        ignore_redundant {bool} -- [description] (default: {True})
        total_maximisation_exploration {int} -- [description] (default: {2000})
        allow_subspace_overlap {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """

    
    
    population_size = population_size
    print(f"*** Optimization algorithm ")
    
    if sampling is None:
        sampling = {
            "ARCHIVE2D": { 
                "ga": 0.5,
                "max": 0.5 },
            "CLOSE": { 
                "ga": 0.25,
                "max": 0.25 },
            "IMP1D": { 
                "ga": 0.15,
                "max": 0.15 },
            "RANDOM": { 
                "ga": 0.1,
                "max": 0.1},
        }
    params = {}
    params["nb_dims"] = nb_dims
    params["hdbscan_min_cluster_size"] = hdbscan_min_cluster_size
    params["population_size"] = population_size
    params["methods"] = [method]
    params["n_clusters"] = n_clusters
    params["min_cluster_size"] = min_cluster_size
    params["loss"] = method
    params["initial_subspace_size"] = 2
    params["truth_methods"] = truth_methods
    params["y"] = truth
    params["selectable_features"] = np.arange(nb_dims)
    params["archive_2d"] = archive_2d
    params["nb_indiv_to_remove_per_round"] = 1
    params["nb_individuals_for_solutions"] = 1
    params["epochs"] = epochs
    params["round_size"] = round_size
    params["debug"] = debug
    params["elitism"] = elitism
    params["max_subspace_size"] = max_subspace_size
    params["sampling_actions"] = ["ARCHIVE2D", "CLOSE", "IMP1D", "RANDOM"]
    params["pca"] = pca
    params["sampling_prob"] = [
        sampling["ARCHIVE2D"]["max"], sampling["CLOSE"]["max"],
        sampling["IMP1D"]["max"], sampling["RANDOM"]["max"]
    ]
    params["meta_features"] = meta_features
    params["score_tolerance"] = score_tolerance
    params["allow_subspace_overlap"] = allow_subspace_overlap
    params["improvement_per_mutation_report"] = improvement_per_mutation_report
    params["clustering"] = clustering
    params["multi_n_clusters"] = multi_n_clusters

    if ignore_redundant == True:
        params["selectable_non_red_features"] = meta_features[
            meta_features["redundant"] == 0]["f"].values
    else:
        params["selectable_non_red_features"] = np.arange(nb_dims)

    params["INS"] = population_size
    params["DEL"] = population_size // 4
    params["REP"] = population_size
    params["CROSS"] = population_size // 2
    params[
        "maximisation_size"] = maximisation_size  # max nb of clustering during maximization phase
    params["maximisation_sizes"] = (maximisation_size * np.array([
        sampling["ARCHIVE2D"]["ga"], sampling["CLOSE"]["ga"],
        sampling["IMP1D"]["ga"], sampling["RANDOM"]["ga"]
    ])).astype(int)
    params["total_maximisation_exploration"] = min(
        int(nb_dims * 0.2), total_maximisation_exploration)
    nb_imp = meta_features[meta_features["1d"] == 1].shape[0]
    print(
        f"Non redundant features {len(params['selectable_non_red_features'])}, orig size {nb_dims}, nb imp : {nb_imp}"
    )

    return params


def run(data, params):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    exploration = np.zeros(data.shape[1])
    solutions = None
    archive = None
    fitness_df = None
    if params["improvement_per_mutation_report"]:
        params["report"] = pd.DataFrame(columns=["indiv", "score", "op", "improvement"])
    for params["epoch"] in tqdm(range(params['epochs'] + 1)):
        if params["epoch"] % (params["round_size"]) == 0:
            solutions, archive  =  run_maximization(
                data, fitness_df, params, archive, exploration, solutions)
            remove_best_subspaces(solutions, params)
            fitness_df, archive, exploration = feature_sampling.create_and_evaluate_population(
                data, params, archive, exploration, fitness_df)
            if params["debug"]:
                print(f"Starting with ... {fitness_df.shape}")
                display(fitness_df)

        fitness_df, exploration, archive = mutations.perform_crossovers(
            data, fitness_df, archive, exploration, params)

        fitness_df, exploration, archive = mutations.perform_mutations(
            data, fitness_df, archive, exploration, params)

        fitness_df = sort_and_purge(fitness_df, params)
        if params["debug"]:
            display(fitness_df)
#             print(exploration_status(exploration, params), Counter(exploration))

    return solutions, archive


def sort_and_purge(fitness_df, params):
    """[summary]

    Arguments:
        fitness_df {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    fitness_df = fitness_df.iloc[fitness_df[[
        'features'
    ]].drop_duplicates().index].sort_values(
        by=params["loss"], ascending=False).reset_index(drop=True)
    if params["elitism"]:
        idx_to_keep = keep_maximal_subspaces(fitness_df.features.values)
        fitness_df = fitness_df.iloc[idx_to_keep].reset_index(drop=True)
    fitness_df = fitness_df.iloc[:params["population_size"]]
    return fitness_df


def keep_maximal_subspaces(featureStrings):
    """
    Remove all split subspaces of inferior size which
    would limit the exploration of other better structures.

    Arguments:
        featureStrings {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    toremove = []
    vals = [set(x.split('_')) for x in featureStrings]
    for i, sp in enumerate(vals):
        for j, sp2 in enumerate(vals[i + 1:]):
            set1 = sp
            set2 = sp2
            if len(set1 - set2) == 0 or len(set2 - set1) == 0:
                assert (sp2 == vals[1 + i + j])
                toremove.append(1 + i + j)
    return np.setdiff1d(np.arange(len(vals)), toremove)


def run_maximization(data, fitness_df, params, archive, exploration,
                      solutions):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        fitness_df {[type]} -- [description]
        params {[type]} -- [description]
        archive {[type]} -- [description]
        exploration {[type]} -- [description]
        solutions {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if fitness_df is None or params["maximisation_size"] == 0:
        return solutions, archive
    if solutions is None:
        solutions = pd.DataFrame()

    for k, individual_str in enumerate(
            fitness_df["features"].
            values[:params["nb_indiv_to_remove_per_round"]]):
        if params["debug"] == True:
            print(f"\n** Maximizing {k}, individual_str {individual_str}")
        best_individual, archive = maximize_individual(
            individual_str, data, params, archive, exploration)
        best_individual = prettify_best_individual(best_individual, data, params)
        # check if the best individual already exists in solution
        if solutions.shape[0] != 0 and str(best_individual["features"]) in [str(ii) for ii in solutions["features"].values]:
            params["epochs"] += params["round_size"]
            continue
        
        solutions = solutions.append(best_individual,
                                         sort=False,
                                         ignore_index=True)
#     values = fitness_df.values
#     values[:params["nb_indiv_to_remove_per_round"]] = solutions.iloc[
#         -params["nb_indiv_to_remove_per_round"]:]
#     fitness_df = pd.DataFrame(columns=fitness_df.columns, data=values)

#     print(f">>> Solutions, archive : {archive.shape}")
    display(solutions)
    return solutions, archive

def prettify_best_individual(best_individual, data, params):
    best_individual["features"] = np.array(best_individual["features"].split('_')).astype(int)
    _, pred = ga_evaluation.clustering_evaluation(best_individual["features"], data, params)
    best_individual["partition"] = pred
    return best_individual

def maximize_individual(individual_str, data, params, archive, exploration):
    """[summary]

    Arguments:
        individual_str {[type]} -- [description]
        data {[type]} -- [description]
        params {[type]} -- [description]
        archive {[type]} -- [description]
        exploration {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    individual = np.array(individual_str.split("_")).astype(int)

    # this will keep the list of features to choose and which didn't lower the score
    if params["allow_subspace_overlap"]:
        params["maximizationFeatures"] = list(np.arange(data.shape[1]))
    else:
        params["maximizationFeatures"] = list(params["selectable_features"])

    max_score = -2
    individual, result, archive, max_score = deletion_round(
        individual,
        data,
        params,
        archive,
        max_score,
        perc_features_to_remove=1)

    while True:
        candidates = feature_sampling.maximization_features(individual, params,
                                                       exploration)
        if len(candidates) == 0:  #everything was explored
            break

        individual, result, archive, found_better, max_score = insertion_round(
            individual, data, params,
            np.array(candidates).astype(int), archive, max_score)

        individual, result, archive, max_score = deletion_round(
            individual,
            data,
            params,
            archive,
            max_score,
            perc_features_to_remove=0.4)

        tabuFeatures = np.setdiff1d(np.arange(data.shape[1]),
                                    params["maximizationFeatures"])

        nb_tabu_featurs = len(tabuFeatures)
        if params["debug"] == True:
            print(f"tabuFeatures ({len(tabuFeatures)}) " +
                  f"{list(tabuFeatures[np.where(tabuFeatures<150)[0]])}")

#         print(f"S={len(individual)}, Tabu = {nb_tabu_featurs}", end = "" )
        if (nb_tabu_featurs + len(individual) > len(
                params["selectable_non_red_features"])
                or  # explored everything
            (
                found_better == False and  # not able to optimize solution
                nb_tabu_featurs >= params['total_maximisation_exploration']) or
            len(individual) > params['max_subspace_size']
           ):
            break
    if params["debug"] == True:
        print(f"Returning maximized individual {result}")
    return result, archive


def insertion_round(individual, data, params, candidates, archive, max_score):
    """[summary]

    Arguments:
        individual {[type]} -- [description]
        data {[type]} -- [description]
        params {[type]} -- [description]
        candidates {[type]} -- [description]
        archive {[type]} -- [description]
        max_score {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    result, archive = ga_evaluation.evaluateIndividual(individual, data,
                                                       params, archive)
    orig_score = result.copy()
    max_score = max(result[params["loss"]], max_score)
    best_indiv = individual.copy()
    #     print(f"Try inserting {len(candidates)}")
    if params["debug"] == True:
        candidates = np.array(candidates)
        print(
            f"ORIG insert: {individual} : {max_score}, size {len(individual)} , candidates "
            #               + f"{list(candidates)}"
            + f"{list(candidates[np.where(candidates<200)[0]])}")

    found_better = False
    added_features = []
    for i in candidates:
        if i not in best_indiv:
            #             if params["fixed_subspace_size"]:
            #                 pivot = best_indiv[np.random.randint(0, len(best_indiv))]
            #                 offspring = [pivot, i]
            offspring = np.sort(np.append(best_indiv, i))
            indiv_scores, archive = ga_evaluation.evaluateIndividual(
                offspring, data, params, archive)
            offspring_score = indiv_scores[params["loss"]]

            delta = offspring_score - max_score
            #             print(f"Added {i} offspring {offspring}, offspring_score {offspring_score}, delta {delta} max {max_score} ")
            if delta > -params["score_tolerance"]:

                max_score = max(offspring_score, max_score)
                result = indiv_scores
                best_indiv = offspring
                found_better = True
                added_features.append(i)
            else:  # bad feature, penalize it
                if i in params["maximizationFeatures"]:
                    params["maximizationFeatures"].remove(i)

    if found_better and params["debug"] == True:
        print(f"Insertion maximization: added {added_features} from #{len(candidates)}" +
              f"=> {best_indiv} ({round(max_score,2)},{len(best_indiv)}f) " +
              f"old : {round(orig_score[params['loss']], 2)} {len(individual)} f")
        
    return best_indiv, result, archive, found_better, max_score


def deletion_round(individual,
                  data,
                  params,
                  archive,
                  max_score,
                  perc_features_to_remove=1):
    """[summary]

    Arguments:
        individual {[type]} -- [description]
        data {[type]} -- [description]
        params {[type]} -- [description]
        archive {[type]} -- [description]
        max_score {[type]} -- [description]

    Keyword Arguments:
        perc_features_to_remove {int} -- [description] (default: {1})

    Returns:
        [type] -- [description]
    """
    record, archive = ga_evaluation.evaluateIndividual(individual, data,
                                                       params, archive)
    max_score = max(record[params["loss"]], max_score)
    if len(individual) == 2:
        return individual, record, archive, max_score

    n_features_to_remove = len(individual)
    if n_features_to_remove > 10:
        n_features_to_remove = int(len(individual) * perc_features_to_remove)

    features_to_remove = np.random.choice(individual,
                                          size=n_features_to_remove,
                                          replace=False)
    #     print(f"Try deleting {len(features_to_remove)}")

    best_indiv = individual.copy()
    removed_features = []
    for i in features_to_remove:
        offspring = [f for f in best_indiv if f != i]
        indiv_scores, archive = ga_evaluation.evaluateIndividual(
            offspring, data, params, archive)
        offspring_score = indiv_scores[params["loss"]]
        delta = offspring_score - max_score
        #         print(f"offspring: {offspring} offspring_score: {offspring_score}, max_score {max_score}, delta {delta}")
        if delta > params["score_tolerance"]:
            if i in params["maximizationFeatures"]:
                params["maximizationFeatures"].remove(i)
            removed_features.append(i)
            max_score = offspring_score
            best_indiv = offspring
            record = indiv_scores
        if len(best_indiv) == 2:
            break
    if params["debug"] == True:
        print(
            f"Deletion maximization: features removed {removed_features}" +
            f"-> {best_indiv} ({max_score},{len(best_indiv)}f); candidates {features_to_remove}"
        )
    return best_indiv, record, archive, max_score


def remove_best_subspaces(solutions, params):
    """[summary]

    Arguments:
        solutions {[type]} -- [description]
        params {[type]} -- [description]
    """
    if solutions is None:
        return
    # remove best features from the next iteration
    best_features = np.unique(np.concatenate(solutions["features"]))
    ## before removing these features, let's try to greedily explore the best indiv to be removed
    params["selectable_non_red_features"] = np.sort(
        np.setdiff1d(params["selectable_non_red_features"], best_features))
    params["selectable_features"] = np.sort(
        np.setdiff1d(params["selectable_features"], best_features))

    if params["debug"]:
        print(f'###\nREMOVING features {np.sort(best_features)}, keep ',
              len(params["selectable_non_red_features"]))



def cluster_population(population, data, n_clusters, method, truth = None, threshold = 0.09,
                      clustering ="gmm"):
    """[summary]

    Arguments:
        population {[type]} -- [description]
        data {[type]} -- [description]
        n_clusters {[type]} -- [description]
        method {[type]} -- [description]

    Keyword Arguments:
        truth {[type]} -- [description] (default: {None})
        threshold {float} -- [description] (default: {0.09})
        clustering {str} -- [description] (default: {"gmm"})

    Returns:
        [type] -- [description]
    """
    params = {}
    params["methods"] = [method]
    params["n_clusters"] = n_clusters
    params["loss"] = method
    params["truth_methods"] = ['ari']
    params["min_cluster_size"] = 4
    params["y"] = truth
    params["clustering"] = clustering
    archive2D = None
    for individual in tqdm(population):
        indiv_scores, archive2D = ga_evaluation.evaluateIndividual(individual, data,  params, archive2D)
    archive2D["feature1"], archive2D["feature2"] = zip(*archive2D["features"].map(
        lambda features_str:np.array(features_str.split('_')).astype(int) ))
    size_before_trim = archive2D.shape[0]
    archive2D = archive2D[archive2D[params["loss"]]>= threshold].sort_values(
        by=params["loss"], ascending=False).reset_index(drop=True)
    print(f"Selecting {len(archive2D)} from {size_before_trim}")
    return archive2D
