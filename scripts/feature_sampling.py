

import sys

import numpy as np
import pandas as pd
import scipy
import scripts.ga_evaluation as ga_evaluation

sys.path.append("..")



def new_random_population(cur_data, params, exploration = None, population_size = None):
    """[summary]

    Arguments:
        cur_data {[type]} -- [description]
        params {[type]} -- [description]

    Keyword Arguments:
        exploration {[type]} -- [description] (default: {None})
        population_size {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    print("RANDOM!!")
    if population_size is None:
        population_size = params["population_size"]
    initial_subspace_size = params["initial_subspace_size"]
    if exploration is None:
        exploration = np.zeros(cur_data.shape[1])
    population = []
    for i in range(population_size):
        
        individual = np.sort(np.random.choice(params["selectable_non_red_features"],size=initial_subspace_size,replace=False, 
                                      p = equal_exploration_probs(exploration, params)))
        exploration= update_exploration(exploration, [individual])
        population.append(individual)
    population = np.stack(population)
    return population, exploration


def new_imp_population(cur_data, params, exploration = None, population_size = None):
    """[summary]

    Arguments:
        cur_data {[type]} -- [description]
        params {[type]} -- [description]

    Keyword Arguments:
        exploration {[type]} -- [description] (default: {None})
        population_size {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    if params["sampling_prob"][-1] ==1: # everything is random
        return new_random_population(cur_data, params, exploration , population_size )

    if population_size is None:
        population_size = params["population_size"]
    initial_subspace_size = params["initial_subspace_size"]
    if exploration is None:
        exploration = np.zeros(cur_data.shape[1])
    meta_features = params["meta_features"]
    features = meta_features[(meta_features["1d"]!= -1)].index.values
    p = get_imp_proba(params)
    population = []
    for i in range(population_size):
        
        individual = np.sort(np.random.choice(params["selectable_non_red_features"],size=initial_subspace_size,replace=False, 
                                      p = p))
        exploration= update_exploration(exploration, [individual])
        population.append(individual)
    population = np.stack(population)
    return population, exploration
def get_imp_proba(params):
    meta_features = params["meta_features"]
    features = meta_features[(meta_features["1d"]!= -1)].index.values

    # important features are 5 times more likely than other features
    n = 5
    p = np.ones(len(params["selectable_non_red_features"]))
    p0 = 1/(len(p) + n*len(features))
    p = p * p0

    p[np.where(np.isin(params["selectable_non_red_features"],features))[0]]=p0 * (n +1)
    # fix floating point pb and requirement for sum (p) to be 1
    if np.sum(p) != 1:
        p[0] += 1-np.sum(p)
    return p

def create_and_evaluate_population(cur_data, params, archive, exploration, fitness_df):
    """[summary]

    Arguments:
        cur_data {[type]} -- [description]
        params {[type]} -- [description]
        archive {[type]} -- [description]
        exploration {[type]} -- [description]
        fitness_df {[type]} -- [description]
    """

    if exploration is None:
        exploration = np.zeros(cur_data.shape[1])
    population_size = params["population_size"]
    population = []

    # Get unexplored individuals from previous round
    previous_round_df = None
    if fitness_df is not None:
        # take the individuals from previous round that have valid features
        previous_idx = []
        for ii, individual_str in enumerate(fitness_df["features"].values):
            individual = np.array(individual_str.split("_")).astype(int)
            # the individual is composed only of remaining features
            if len(np.setdiff1d(individual, params["selectable_non_red_features"])) ==0 :
                previous_idx.append(ii)
        previous_round_df = fitness_df.iloc[previous_idx]
        if params["debug"] == True:
            print(f"Adding from previous round {previous_round_df.shape}")
        population_size = population_size - len(previous_round_df)
              
    # CASE 2: start form 2D archive if subspace > 2
    if params['archive_2d'] is not None:
        round_nb = params["epoch"]//params["round_size"] 
#         unexplored= np.arange(round_nb*population_size, population_size*(round_nb+1))
        
        archive_2d = params['archive_2d'][
            (params['archive_2d'].feature1.isin(params["selectable_non_red_features"])) &
            (params['archive_2d'].feature2.isin(params["selectable_non_red_features"])) 
        ].reset_index(drop = True).iloc[:population_size]

        if len(archive_2d) > 0:
            population.extend(archive_2d['features'].values)
            population = [np.array(features_str.split('_')).astype(int) for features_str in population]
            exploration= update_exploration(exploration, population)
            print(f"Selecting {archive_2d.shape} from archive")

    # CASE 4: when the features were well explored and not enough left, add random selections
    if len(population)< population_size:
#         print("Generating random population...")
        population_size = population_size - len(population)
        # create random individuals
#         remanining_population, exploration = new_random_population(
        remanining_population, exploration = new_imp_population(
                    cur_data,
                    params,
                    exploration=exploration,
                    population_size = population_size)
        print(f"adding {population_size} random population")
        population.extend(remanining_population)
    fitness_df = pd.DataFrame()
    if params["debug"]:
        print(f"Evaluating {len(population)} individuals ... ")
    for individual in population:
        indiv_scores, archive = ga_evaluation.evaluateIndividual(individual ,cur_data, params, archive)
        fitness_df = fitness_df.append(indiv_scores, ignore_index = True, sort = False)
    if previous_round_df is not None:
        fitness_df = pd.concat([fitness_df, previous_round_df], ignore_index = True, sort = False)

    fitness_df = fitness_df.drop_duplicates().sort_values(by=params["loss"], 
           ascending=False).reset_index(drop=True).iloc[:params["population_size"]]

#     if fitness_df.shape[0] <params["population_size"]:
#         raise Exception('Wrong initial population size!')
    return fitness_df, archive, exploration


def equal_exploration_probs(e, params = None):
    """[summary]

    Arguments:
        e {[type]} -- [description]

    Keyword Arguments:
        params {[type]} -- [description] (default: {None})

    Returns:
        [type] -- [description]
    """
    if params is not None:
        e = e[params["selectable_non_red_features"]]
    if np.sum(e) == 0: # no choice has been made, everything is equiprobable
        return np.ones_like(e)/len(e)
    e = np.abs(e-e.max())
    p = scipy.special.softmax(e)
    return p


def update_exploration(exploration, population):
    """[summary]

    Arguments:
        exploration {[type]} -- [description]
        population {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    for i in range(len(population)):
        exploration[population[i].astype(int)]+=1 
    return exploration


def choose_unexplored_features(current, nb_mutations, exploration, params, choose_from = None):
    """[summary]

    Arguments:
        current {[type]} -- [description]
        nb_mutations {[type]} -- [description]
        exploration {[type]} -- [description]
        params {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    if choose_from is None:
        choose_from = np.sort(np.setdiff1d(params["selectable_non_red_features"], current))
    else:
        choose_from = np.sort(np.setdiff1d(choose_from, current))
    if len(choose_from) ==0:
        choose_from = np.sort(np.setdiff1d(np.arange(params['nb_dims']), current))
                              
    p = equal_exploration_probs(exploration[choose_from])
    mutation_idx = np.random.choice(np.arange(len(choose_from)),
                                    size=min(nb_mutations, len(choose_from)),
                                    replace=False,
                                    p=p)
    mut_features = choose_from[mutation_idx]
    return np.sort(mut_features)


def sample_features(current,
                           nb_mutations,
                           exploration,
                           params,
                           source=None,
                           choose_from = None
                          ):
    """[summary]

    Arguments:
        current {[type]} -- [description]
        nb_mutations {[type]} -- [description]
        exploration {[type]} -- [description]
        params {[type]} -- [description]

    Keyword Arguments:
        source {[type]} -- [description] (default: {None})
        choose_from {[type]} -- [description] (default: {None})
    """
    if choose_from is None:
        choose_from = np.sort(np.setdiff1d(params["selectable_non_red_features"], current))
        
    features = []
    meta_features = params["meta_features"]
    if source is None:
        # ["RANDOM", "ARCHIVE2D", "CLOSE", "IMP1D", "OUTLIER"]
        source = np.random.choice(params["sampling_actions"],
                                  size=None,
                                  p=params["sampling_prob"])
    if source == "ARCHIVE2D" and params['archive_2d'] is not None:
        features = np.setdiff1d(
            np.unique(params['archive_2d']
                      [(params['archive_2d'].feature1.isin(current)) |
                       (params['archive_2d'].feature2.isin(current))][[
                           'feature1', 'feature2'
                       ]].values.ravel()), current)
        
    if source == "CLOSE":
        features = np.unique(meta_features[(meta_features.index.isin(current))][
            ["f1", "f2", "f3"]].values.ravel())
        current_clusters =meta_features[(meta_features.index.isin(current))]["clusters"].unique()
        cluster_features = meta_features[meta_features["clusters"].isin(current_clusters)]["f"]
        features = np.unique(np.concatenate([features, cluster_features]))
        
    if source == "OUTLIER":
        features = meta_features[meta_features["type"] == "outlier"].index.values
        
    if source == "IMP1D":
        features = meta_features[~meta_features.index.isin(current) &
             (meta_features["1d"]!= -1)].index.values
        
    # remove current element
    features = np.setdiff1d(features, current)
    features = np.intersect1d(features, choose_from)
    if len(features) >0:
        p = equal_exploration_probs(exploration[features.astype(int)])
        try:
            features = np.random.choice(features,size = min(nb_mutations, len(features)) 
                                        , replace=False, p = p)
        except ValueError:
            features = np.random.choice(features,size = min(nb_mutations, len(features)) 
                                        , replace=True, p = p)

    # Default to RANDOM
    if len(features) < nb_mutations:  # still didn't find good combinations
        exploFeatures = list(
            choose_unexplored_features(current, nb_mutations - len(features),
                                        exploration, params, choose_from))
        features = list(features)
        features.extend(exploFeatures)
        source = "RANDOM"
#     if params["debug"]:
#         print(f"{source} ", end= '')
    return np.sort(features)


def maximization_features(individual, params, exploration):
    """[summary]

    Arguments:
        individual {[type]} -- [description]
        params {[type]} -- [description]
        exploration {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    features = []
    for i, action in enumerate(params["sampling_actions"]):
        nb_candidates = params["maximisation_sizes"][i]
        if action == "RANDOM":
            nb_candidates = params["maximisation_size"] - len(np.unique(features))
        cur_features = sample_features(individual,
                           nb_candidates,
                           exploration,
                           params,
                           source=action,
                           choose_from = params["maximizationFeatures"]
                          )
#         ints = np.setdiff1d(cur_features, params["maximizationFeatures"])
#         if len(ints) >0:
#             print(f"**** {action}, {len(ints)}")
        features.extend(cur_features)
    features = np.unique(features)
    return features
