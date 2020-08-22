import os

import numpy as np
import pandas as pd
import scripts.cnn_models as cnn_models
import tensorflow as tf
from keras.models import load_model
from sklearn import preprocessing


def run(data,
        n_clusters,
        meta_features,
        model_file,
        trim_limit=10000,
        theta=0.1,
        add_close_population=False,
        exploration_factor = 3):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        n_clusters {[type]} -- [description]
        meta_features {[type]} -- [description]
        model_file {[type]} -- [description]

    Keyword Arguments:
        trim_limit {int} -- [description] (default: {10000})
        theta {float} -- [description] (default: {0.1})
        add_close_population {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    print(f"*** Exploring 2D feature space with NN ...")
    data = preprocessing.MinMaxScaler().fit_transform(data)

    digitized_data_1d = cnn_models.digitize(data)
    if model_file is None or os.path.isfile(model_file) == False:
        print(f"Please provide a trained model file at path {model_file}")
        return None, None
    model12d = load_model(model_file)
    n_total = 0
    result = None
    result, n = explore_close_imp_pairs(
        digitized_data_1d,
        n_clusters,
        meta_features,
        result,
        model12d,
        trim=True,
        theta=theta,
        add_close_population=add_close_population,
        trim_limit=trim_limit,
        exploration_factor = exploration_factor)

    n_total += n
    result, n = explore_imp_imp_pairs(
        digitized_data_1d,
        n_clusters,
        meta_features,
        result,
        model12d,
        trim=True,
        theta=theta,
        add_close_population=add_close_population,
        trim_limit=trim_limit,
        exploration_factor = exploration_factor)
    n_total += n

    result, n = explore_non_imp_pairs(
        digitized_data_1d,
        n_clusters,
        meta_features,
        result,
        model12d,
        trim=True,
        theta=theta,
        add_close_population=add_close_population,
        trim_limit=trim_limit,
        exploration_factor = exploration_factor)

    n_total += n

    result, n = explore_mixed_pairs(digitized_data_1d,
                                    n_clusters,
                                    meta_features,
                                    result,
                                    model12d,
                                    trim=True,
                                    theta=theta,
                                    add_close_population=add_close_population,
                                    trim_limit=trim_limit, 
                                    exploration_factor = exploration_factor)
    n_total += n

    result = result.drop_duplicates().sort_values(by="pred", ascending=False)
    result = result.rename(columns={"f1": "feature1", "f2": "feature2"})
    print(
        f"Returning {result.shape}, explored a total of {n_total} feature pairs"
    )
    result["features"] = result["feature1"].astype(
        str) + "_" + result["feature2"].astype(str)
    result["feature1"] = result["feature1"].astype(int)
    result["feature2"] = result["feature2"].astype(int)
    return result, n_total


def trim_results(current_result,
                 result,
                 max_size,
                 theta=0.09,
                 trim_limit=10000):
    """[summary]

    Arguments:
        current_result {[type]} -- [description]
        result {[type]} -- [description]
        max_size {[type]} -- [description]

    Keyword Arguments:
        theta {float} -- [description] (default: {0.09})
        trim_limit {int} -- [description] (default: {10000})

    Returns:
        [type] -- [description]
    """
    # trim data
    existing_values = (result["f1"].astype(str) + "_" +
                       result["f2"].astype(str)).values
    current_values = (current_result["f1"].astype(str) + "_" +
                      current_result["f2"].astype(str)).values
    idx = np.where(~np.isin(current_values, existing_values))
    current_result = current_result.iloc[idx]
    current_result = current_result[
        current_result["pred"] > theta].reset_index(drop=True)
    if max_size is not None:
        max_size = min(max_size, trim_limit)
        if current_result.shape[0] > max_size:
            print(f"trimming {current_result.shape[0]} to {max_size}")
            current_result = current_result.iloc[:max_size]
    result = pd.concat([current_result, result], ignore_index=True)  #
    return result


def explore_close_imp_pairs(digitized_data_1d,
                            n_clusters,
                            meta_features,
                            result,
                            model12d,
                            trim=True,
                            theta=0.09,
                            trim_limit=10000,
                            add_close_population=False,
                            exploration_factor = 3):
    if result is None:
        result = pd.DataFrame(columns=["f1", "f2", "pred"])
    ## Create population
    current_result = pd.DataFrame(columns=["f1", "f2", "pred"])

    imp_close_population1 = meta_features[(meta_features["relevance"] != 0) & (
        meta_features["redundant"] == 0)][["f", "f1"]].values
    imp_close_population2 = meta_features[(meta_features["relevance"] != 0) & (
        meta_features["redundant"] == 0)][["f", "f2"]].values
    imp_close_population3 = meta_features[(meta_features["relevance"] != 0) & (
        meta_features["redundant"] == 0)][["f", "f3"]].values

    imp_close_population = np.sort(
        np.concatenate([
            imp_close_population1, imp_close_population2, imp_close_population3
        ]))

    imp_close_population = np.unique(imp_close_population, axis=0)

    imp_close_population = imp_close_population[np.where(
        imp_close_population[:, 1] - imp_close_population[:, 0] != 0)[0]]
    imp_close_population = imp_close_population.astype(int)
    n = len(imp_close_population)
    res = cnn_models.predict(model12d, imp_close_population, digitized_data_1d,
                             n_clusters)

    current_result = pd.concat(
        [current_result, res],
        ignore_index=True)  #.sort_values(by="pred", ascending = False)
    if add_close_population and res.shape[0] != 0:
        new_close_population = create_new_population(res, meta_features)
        n += (len(new_close_population))
        res = cnn_models.predict(model12d, new_close_population,
                                 digitized_data_1d, n_clusters)
        res = res[res["pred"] > theta]
        current_result = pd.concat([current_result, res], ignore_index=True)  #
    max_size = meta_features[(meta_features["redundant"] == 0)].shape[0] // 2
#     print(f"Max {result['pred'].max()}")
    result = trim_results(current_result,
                          result,
                          max_size,
                          theta=theta,
                          trim_limit=trim_limit)
    print(f"handle_close_important {result.shape}, total {n}, {result['pred'].max()}")
    return result, n


def explore_imp_imp_pairs(digitized_data_1d,
                          n_clusters,
                          meta_features,
                          result,
                          model12d,
                          trim=True,
                          theta=0.09,
                          trim_limit=10000,
                          add_close_population=False,
                          exploration_factor=3):
    if result is None:
        result = pd.DataFrame(columns=["f1", "f2", "pred"])

    current_result = pd.DataFrame(columns=["f1", "f2", "pred"])

    relevant_features = meta_features[(meta_features["relevance"] != 0) & (
        meta_features["redundant"] == 0)]["f"].values

    population = make_random_population(relevant_features, meta_features,
                                        len(relevant_features) * exploration_factor*2)
    print(
        f"relevant_features {len(relevant_features)} => computing {len(population)} "
    )
    n = len(population)
    res = cnn_models.predict(model12d, population, digitized_data_1d,
                             n_clusters)
    res = res[res["pred"] > theta]
    current_result = pd.concat(
        [current_result, res],
        ignore_index=True)  #.sort_values(by="pred", ascending = False)
    if add_close_population and res.shape[0] != 0:
        new_close_population = create_new_population(
            res.sort_values(by="pred", ascending=False).iloc[:1000],
            meta_features)
        n += len(new_close_population)
        res = cnn_models.predict(model12d, new_close_population,
                                 digitized_data_1d, n_clusters)
        if res is not None:
            current_result = pd.concat(
                [current_result, res],
                ignore_index=True)  #.sort_values(by="pred", ascending = False)
    max_size = meta_features[(meta_features["redundant"] != 0)].shape[0]
#     print(f"Max {result['pred'].max()}")
    result = trim_results(current_result,
                          result,
                          max_size,
                          theta=theta,
                          trim_limit=trim_limit)
    print(f"handle_important_features {result.shape},  total {n}, {result['pred'].max()}")
    return result, n


def explore_non_imp_pairs(digitized_data_1d,
                          n_clusters,
                          meta_features,
                          result,
                          model12d,
                          trim=True,
                          theta=0.09,
                          trim_limit=10000,
                          add_close_population=False,
                          exploration_factor=3):
    if result is None:
        result = pd.DataFrame(columns=["f1", "f2", "pred"])
    current_result = pd.DataFrame(columns=["f1", "f2", "pred"])

    irrelevant_features = meta_features[(meta_features["relevance"] == 0) & (
        meta_features["redundant"] == 0)]["f"].values
    population = make_random_population(irrelevant_features, meta_features,
                                        len(irrelevant_features) * exploration_factor)
    print(
        f"irrelevant_features {len(irrelevant_features)} => computing {len(population)}"
    )
    n = len(population)
    res = cnn_models.predict(model12d, population, digitized_data_1d,
                             n_clusters)
    res = res[res["pred"] > theta]
    current_result = pd.concat([current_result, res], ignore_index=True)

    if add_close_population and res is not None and res.shape[0] > 0:
        new_close_population = create_new_population(
            res.sort_values(by="pred", ascending=False).iloc[:1000],
            meta_features)
        n += len(new_close_population)
        res = cnn_models.predict(model12d, new_close_population,
                                 digitized_data_1d, n_clusters)
        if res is not None and res.shape[0] > 0:
            current_result = pd.concat(
                [current_result, res],
                ignore_index=True)  #.sort_values(by="pred", ascending = False)

    max_size = meta_features[(meta_features["redundant"] == 0)].shape[0] // 2
#     print(f"Max {result['pred'].max()}")
    result = trim_results(current_result,
                          result,
                          max_size,
                          theta=theta,
                          trim_limit=trim_limit)
    print(f"handle_not_important_features {result.shape}, total {n}, {result['pred'].max()}")
    return result, n


def explore_mixed_pairs(digitized_data_1d,
                        n_clusters,
                        meta_features,
                        result,
                        model12d,
                        trim=True,
                        theta=0.09,
                        trim_limit=10000,
                        add_close_population=False,
                        exploration_factor = 3):
    """[summary]

    Arguments:
        digitized_data_1d {[type]} -- [description]
        n_clusters {[type]} -- [description]
        meta_features {[type]} -- [description]
        result {[type]} -- [description]
        model12d {[type]} -- [description]

    Keyword Arguments:
        trim {bool} -- [description] (default: {True})
        theta {float} -- [description] (default: {0.09})
        trim_limit {int} -- [description] (default: {10000})
        add_close_population {bool} -- [description] (default: {False})

    Returns:
        [type] -- [description]
    """
    if result is None:
        result = pd.DataFrame(columns=["f1", "f2", "pred"])
    current_result = pd.DataFrame(columns=["f1", "f2", "pred"])

    all_features = meta_features[(meta_features["redundant"] == 0)]["f"].values
    population = make_random_population(all_features, meta_features,
                                        len(all_features) * exploration_factor)
    print(
        f"handle_all_features {len(all_features)} => computing {len(population)}"
    )
    n = len(population)
    res = cnn_models.predict(model12d, population, digitized_data_1d,
                             n_clusters)
    res = res[res["pred"] > theta]

    current_result = pd.concat(
        [current_result, res],
        ignore_index=True)  #.sort_values(by="pred", ascending = False)

    if add_close_population and res.shape[0] != 0:
        new_close_population = create_new_population(
            res.sort_values(by="pred", ascending=False).iloc[:1000],
            meta_features)
        n += len(new_close_population)
        res = cnn_models.predict(model12d, new_close_population,
                                 digitized_data_1d, n_clusters)
        if res is not None and res.shape[0] > 0:
            current_result = pd.concat(
                [current_result, res],
                ignore_index=True)  #.sort_values(by="pred", ascending = False)
    max_size = meta_features[(meta_features["redundant"] == 0)].shape[0]
#     print(f"Max {result['pred'].max()}")
    result = trim_results(current_result,
                          result,
                          max_size,
                          theta=theta,
                          trim_limit=trim_limit)
    print(f"handle_all_features {result.shape},  total {n}, {result['pred'].max()}")
    return result, n


def get_relevant_population(relevant_features, centroid_features,
                            n_relevant_clusters):
    relevant_population = np.zeros(
        (len(relevant_features) * n_relevant_clusters, 2))

    relevant_population[:, 0] = np.repeat(centroid_features,
                                          len(relevant_features))
    relevant_population[:, 1] = np.tile(relevant_features, n_relevant_clusters)
    relevant_population = np.sort(relevant_population)
    relevant_population = relevant_population[np.where(
        relevant_population[:, 1] - relevant_population[:, 0] != 0)[0]]
    relevant_population = np.unique(relevant_population, axis=0)
    relevant_population = relevant_population.astype(int)

    return relevant_population


def create_new_population(res, meta_features):
    if res.shape[0] == 0:
        return None
    new_close_population = np.zeros((res.shape[0], 2))
    new_close_population[:, 0] = pd.merge(res,
                                          meta_features[["f", "f1"]],
                                          left_on="f1",
                                          right_on="f",
                                          how="left")["f1_y"].values

    new_close_population[:, 1] = pd.merge(res,
                                          meta_features[["f", "f1"]],
                                          left_on="f2",
                                          right_on="f",
                                          how="left")["f1_y"].values

    new_close_population = np.sort(new_close_population)
    new_close_population = new_close_population[np.where(
        new_close_population[:, 1] - new_close_population[:, 0] != 0)[0]]
    new_close_population = np.unique(new_close_population, axis=0)
    new_close_population = new_close_population.astype(int)
    return new_close_population


def make_random_population(relevant_features, meta_features, n):

    np.random.seed(0)
    idx1 = relevant_features[np.random.randint(0, len(relevant_features),
                                               n)].reshape(-1, 1)
    idx2 = relevant_features[np.random.randint(0, len(relevant_features),
                                               n)].reshape(-1, 1)

    population = np.concatenate([idx1, idx2], axis=1)

    population = np.sort(population)
    population = population[np.where(
        population[:, 1] - population[:, 0] != 0)[0]]
    population = np.unique(population, axis=0)
    population = population.astype(int)
    return population
