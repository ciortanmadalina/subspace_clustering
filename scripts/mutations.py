
import itertools
import sys

import numpy as np
import pandas as pd
import scripts.feature_sampling as feature_sampling
import scripts.ga_evaluation as ga_evaluation
from tqdm import tqdm

sys.path.append("..")


########## CROSS OVERS ##################################
def crossover(i1, i2):
    """[summary]

    Arguments:
        i1 {[type]} -- [description]
        i2 {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    union = np.unique(np.concatenate([i1, i2]))
    while True:
        size = np.random.randint(min(len(i1), len(i2)) + 1, len(union) + 1)
        offspring = np.sort(np.random.choice(union, size=size, replace=False))
        if np.array_equal(offspring, i1) == False and np.array_equal(
                offspring, i2) == False:
            break

    return offspring


def perform_crossovers(data, fitness_df, archive, exploration, params):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        fitness_df {[type]} -- [description]
        archive {[type]} -- [description]
        exploration {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    cross_over_population = pd.DataFrame()
    fs = params["loss"]
    # select candidates for cross over as a mixture between the best and random exploration
    combinations = np.array(
        list(itertools.combinations(np.arange(len(fitness_df)), 2)))
    combinations_best_idx = np.random.choice(np.arange(len(combinations)),
                                             size=params["CROSS"],
                                             replace=True)

    for i, j in combinations[combinations_best_idx]:

        parent1 = fitness_df.iloc[i]
        parent_features1 = np.array(parent1['features'].split('_')).astype(int)
        parent2 = fitness_df.iloc[j]
        parent_features2 = np.array(parent2['features'].split('_')).astype(int)
        # one individual is contained in the other
        if len(np.setdiff1d(parent_features1, parent_features2)) == 0 or len(
                np.setdiff1d(parent_features2, parent_features1)) == 0:
            continue
        offspring = crossover(parent_features1, parent_features2)
        indiv_scores, archive = ga_evaluation.evaluateIndividual(
            offspring, data, params, archive)

        if params["debug"]:
            print(
                f'{parent_features1}({round(parent1[fs],3)}) + {parent_features2}({round(parent2[fs], 3)})->{offspring} ({round(indiv_scores[fs], 3)})'
            )
        if params["improvement_per_mutation_report"]:
            params["report"].loc[params["report"].shape[0]] = [
                indiv_scores["features"], indiv_scores[params['loss']],
                f"CROSS",
                (indiv_scores[params['loss']] > max(
                    parent1[fs], parent2[fs]))
            ]

        cross_over_population = cross_over_population.append(indiv_scores,
                                                             ignore_index=True,
                                                             sort=False)
        exploration = feature_sampling.update_exploration(exploration, [offspring])

    fitness_df = pd.concat([fitness_df, cross_over_population],
                           ignore_index=True,
                           sort=False).drop_duplicates().sort_values(
                               by=params["loss"],
                               ascending=False).reset_index(drop=True)
    return fitness_df, exploration, archive


########## MUTATIONS ##################################
def insert_mutation(current, exploration, params):
    """[summary]

    Arguments:
        current {[type]} -- [description]
        exploration {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    nb_mutations = 1
    mut_features = feature_sampling.sample_features(current,
                                                      nb_mutations,
                                                      exploration,
                                                      params,
                                                      source=None)
    mutation = np.sort(np.append(current, mut_features))
    return mutation, mut_features


def replace_mutation(current, exploration, params):
    """[summary]

    Arguments:
        current {[type]} -- [description]
        exploration {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    nb_mutations = np.random.choice([1, 2], size=None, p=[0.1, 0.9])
    mut_features = feature_sampling.sample_features(current,
                                                      nb_mutations,
                                                      exploration,
                                                      params,
                                                      source=None)

    pivot = current[np.random.randint(0, len(current))]
    mutation = np.append(current, mut_features)
    mutation = np.sort(np.setdiff1d(mutation, [pivot]))
    return mutation, mut_features


def delete_mutation(current):
    """[summary]

    Arguments:
        current {[type]} -- [description]
        exploration {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    deletion_idx = np.random.randint(0, len(current))
    mut_features = [np.nan]
    mutation = current[np.setdiff1d(np.arange(len(current)), [deletion_idx])]
    return mutation, mut_features


def mutation_operation(operation, parent_features, exploration, params):
    """[summary]

    Arguments:
        operation {[type]} -- [description]
        parent_features {[type]} -- [description]
        exploration {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if operation == "INS":
        offspring, mut_features = insert_mutation(parent_features,
                                                    exploration, params)
    if operation == "REP":
        offspring, mut_features = replace_mutation(parent_features,
                                                      exploration, params)
    if operation == "DEL":
        if len(parent_features) > 2:
            offspring, mut_features = delete_mutation(parent_features)
        else:
            offspring, mut_features = None, None
    return offspring, mut_features


def perform_mutations(data, fitness_df, archive, exploration, params):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        fitness_df {[type]} -- [description]
        archive {[type]} -- [description]
        exploration {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    # compute mutations
    mutation_scores = pd.DataFrame()
    mutations = ["INS", "REP", "DEL"]
    for operation in mutations:
        if params[operation] == 0:
            continue
        mutation_indexes = np.random.choice(np.arange(len(fitness_df)),
                                            size=params[operation],
                                            replace=True)
        for mutation_idx in mutation_indexes:
            parent = fitness_df.iloc[mutation_idx]
            parent_features = np.array(
                parent['features'].split('_')).astype(int)
            previous_score = parent[params["loss"]]
            offspring, mut_features = mutation_operation(
                operation, parent_features, exploration, params)
            if offspring is None:  # the operation was forbidden
                continue
            indiv_scores, archive = ga_evaluation.evaluateIndividual(
                offspring, data, params, archive)
            if params["debug"]:
                print(
                    f"{operation} {parent_features}-> {offspring}, scores : " +
                    f" {round(previous_score, 3)}-> {round(indiv_scores[params['loss']], 3)}"
                )
            if params["improvement_per_mutation_report"]:
                params["report"].loc[params["report"].shape[0]] = [
                    indiv_scores["features"],
                    indiv_scores[params['loss']], f"mutation_{operation}",
                    (indiv_scores[params['loss']] > previous_score)
                ]
            mutation_scores = mutation_scores.append(indiv_scores,
                                                     ignore_index=True,
                                                     sort=False)
            exploration = feature_sampling.update_exploration(
                exploration, [offspring])

    fitness_df = pd.concat([fitness_df, mutation_scores],
                           ignore_index=True,
                           sort=False).drop_duplicates().sort_values(
                               by=params["loss"],
                               ascending=False).reset_index(drop=True)
    return fitness_df, exploration, archive
