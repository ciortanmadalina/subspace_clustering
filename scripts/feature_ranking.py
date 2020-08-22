import itertools
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scripts.statistics as statistics
import seaborn as sns
import xgboost as xgb
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn import preprocessing
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

sys.path.append("..")


def rank_features(data,
                  nb_bins=20,
                  rank_threshold=80,
                  z_file=None,
                  metric='euclidean',
                  redundant_threshold=1,
                  nb_consensus_methods = 3):
    """
    Performs unsupervised feature selections
    Arguments:
        data {[type]} -- [description]

    Keyword Arguments:
        nb_bins {int} -- [description] (default: {20})
        rank_threshold {float} -- [description] (default: {80})
        z_file {[type]} -- [description] (default: {None})
    """
    print(f"*** Computing 1D feature ranking ...")
    t1 = time.time()
    mad = statistics.mean_abs_difference(data)
    dispersion = statistics.dispersion(data)
    mean_median = statistics.mean_median(data)
    amam = statistics.amam(data)
    spec = statistics.spec_scores(data)

    t2 = time.time()
    print(f"Dispersion tests took {round(t2-t1, 2)} sec")

    entropy = np.array([
        statistics.compute_entropy(data[:, i], nb_bins=20)
        for i in range(data.shape[1])
    ])
    t3 = time.time()
    print(f"Entropy computation {round(t3-t2, 2)} sec")

    # compute 3 nearest neighbors and get feature type
    nbrs = NearestNeighbors(n_neighbors=4, algorithm='brute',
                            metric=metric).fit(data.T)
    distances, indices = nbrs.kneighbors(data.T)

    meta_features = pd.concat([
        pd.DataFrame(data=distances[:, 1:], columns=["d1", "d2", "d3"]),
        pd.DataFrame(data=indices[:, 1:], columns=["f1", "f2", "f3"])
    ],
                              axis=1)
    #     meta_features = pd.DataFrame(index = np.arange(data.shape[1]))
    t4 = time.time()
    print(f"KNN computation {round(t4-t3, 2)} sec")

    meta_features["f"] = meta_features.index
    meta_features["mad"] = mad
    meta_features["dispersion"] = dispersion
    meta_features["amam"] = amam
    meta_features["mean_median"] = mean_median
    meta_features["entropy"] = entropy
    meta_features["spec"] = -spec
    #     meta_features["kstest"] = kstest
    meta_features["uniform"] = (meta_features["entropy"] > 0.95).astype(int)
    features_to_scale = ["mad", "dispersion", "amam", "mean_median", "spec"]
    scaled_values = preprocessing.MinMaxScaler().fit_transform(
        meta_features[meta_features["uniform"] == 0][features_to_scale].values)

    thresholds = np.percentile(scaled_values, rank_threshold, axis=0)
    scaled_values = (scaled_values > thresholds).astype(int)

    meta_features["rank_mad"] = -1
    meta_features["rank_dispersion"] = -1
    meta_features["rank_amam"] = -1
    meta_features["rank_mean_median"] = -1
    meta_features["rank_spec"] = -1

    meta_features.loc[meta_features[meta_features["uniform"] == 0].index, [
        "rank_mad", "rank_dispersion", "rank_amam", "rank_mean_median",
        "rank_spec"
    ]] = scaled_values

    meta_features["relevance"] = meta_features[[
        "rank_mad", "rank_dispersion", "rank_amam", "rank_mean_median",
        "rank_spec"
    ]].sum(axis=1)

    t5 = time.time()
    print(f"Sorting and thresholds {round(t5-t4, 2)} sec")
    if z_file is not None:
        print("Loading clustering from file")
        Z = np.load(z_file)
    else:
        print("Performing hierarchical clustering...")
        Z = linkage(data.T, method='complete', metric=metric)

    pred = fcluster(Z, redundant_threshold, criterion='distance')
    redundant = np.zeros_like(pred)

    t6 = time.time()
    print(f"Hierarchical clustering {round(t6-t5, 2)} sec")
    for c in np.unique(pred):
        idx = np.where(pred == c)[0]
        if len(idx) > 1:
            redundant[idx] = 1
            # select the features with the highest relevance score
            representative_features = meta_features[meta_features["f"].isin(
                idx)].sort_values(by="relevance",
                                  ascending=False)["f"].values[:2]
            redundant[representative_features] = 0

        else:
            redundant[idx] = 0
    t7 = time.time()
    print(f"Handle redundant features {round(t7-t6, 2)} sec")

    meta_features["clusters"] = pred
    meta_features["redundant"] = redundant

    meta_features["1d"] = meta_features["relevance"].apply(lambda x: 1
                                                   if x >= nb_consensus_methods else -1)
    t8 = time.time()
    print(f'Returning {meta_features["redundant"].value_counts().get(1, 0)} redundant features ' +
          f'and  {meta_features["1d"].value_counts().get(1, 0)} important features')
    return meta_features


def get_closest_features(data):
    """[summary]

    Arguments:
        data {[type]} -- [description]

    Returns:
        [type] -- [description]
    """
    closest_df = pd.DataFrame(columns=["closest1", "closest2"])
    all_features = np.arange(data.shape[1])
    for i in tqdm(all_features):
        other_features = np.setdiff1d(all_features, i)
        distances = np.array([
            pdist(data[:, [i, j]].T, metric='euclidean')[0]
            for j in other_features
        ])
        sorted_idx = np.argsort(distances)
        closest_1 = other_features[sorted_idx[0]]
        closest_2 = other_features[sorted_idx[1]]
        closest_df.loc[closest_df.shape[0]] = [closest_1, closest_2]
    return closest_df


def supervised_feature_ranking(data, truth, nbTopFeatures=5, random_state=42):
    def getScoresAsDf(fi):
        keys = list(fi.keys())
        values = list(fi.values())
        fiDf = pd.DataFrame()
        fiDf['score'] = values
        fiDf['feature'] = keys
        fiDf['feature'] = fiDf['feature'].astype(int)
        fiDf = fiDf.sort_values(by="score", ascending=False)
        fiDf = fiDf.set_index('feature').reindex(np.arange(
            data.shape[1])).fillna(0).reset_index()
        fiDf[['score'
              ]] = preprocessing.MinMaxScaler().fit_transform(fiDf[['score']])
        return fiDf

    dtrain = xgb.DMatrix(data=pd.DataFrame(data), label=truth)
    dtest = xgb.DMatrix(data=pd.DataFrame(data))

    params = {
        'max_depth': 6,
        'objective':
        'multi:softmax',  # error ga_evaluation for multiclass training
        'num_class': len(np.unique(truth)),
        'n_gpus': 0,
        'random_state': random_state
    }

    model = xgb.train(params, dtrain)

    pred = model.predict(dtest)

    #     print('test ARI=', adjusted_rand_score(y_test, pred))

    fiWeightDf = getScoresAsDf(model.get_score(importance_type='weight'))
    fiGainDf = getScoresAsDf(model.get_score(importance_type='gain'))

    fiDf = pd.merge(fiWeightDf,
                    fiGainDf,
                    left_on='feature',
                    right_on='feature')

    fiDf['score'] = fiDf[['score_x', 'score_y']].mean(axis=1).values
    ranked_features = fiDf.sort_values(
        by='score', ascending=False)['feature'].values[:nbTopFeatures]
    return ranked_features


def analyze_measure_results(
    meta_features,
    measures=['mad', 'dispersion', 'amam', 'mean_median', 'entropy', 'spec'],
    ranks=[
        'rank_mad', 'rank_dispersion', 'rank_amam', 'rank_mean_median',
        'rank_spec'
    ]):
    plt.figure()
    ax = plt.gca()
    plt.title("Correlations between scores")
    sns.heatmap(meta_features[measures].corr().round(2),
                annot=True,
                fmt=".2g",
                cmap="coolwarm",
                ax=ax)
    plt.tight_layout()

    # agreement analysis
    agreement = np.zeros((len(ranks), len(ranks)))
    disagreement = np.zeros((len(ranks), len(ranks)))
    for i, j in itertools.combinations(np.arange(len(ranks)), 2):
        counts = meta_features[[ranks[i], ranks[j]]].sum(axis=1).value_counts()
        agreement[i, j] = counts.get(2, 0)
        agreement[j, i] = counts.get(2, 0)

        disagreement[i, j] = counts.get(1, 0)
        disagreement[j, i] = counts.get(1, 0)
    for i in range(len(ranks)):
        counts = meta_features[[ranks[i], ranks[i]]].sum(axis=1).value_counts()
        agreement[i, i] = counts.get(2, 0)
        disagreement[i, i] = counts.get(1, 0)
    labels = ["MAD", "Dispersion", "AMAM", "Mean Median", "SPEC"]
    agreement = pd.DataFrame(data=agreement, columns=labels,
                             index=labels).astype(int)
    disagreement = pd.DataFrame(data=disagreement, columns=labels,
                                index=labels).astype(int)

    plt.figure(figsize=(12, 5))
    ax = plt.subplot(121)
    plt.title("Agreement between measures\n Number of commonly chosen features")
    sns.heatmap(agreement, annot=True, ax=ax, fmt="d", cmap="coolwarm")

    ax = plt.subplot(122)
    plt.title("Disagreement  between measures\n Number of feature choice mismatches")
    sns.heatmap(disagreement, annot=True, ax=ax, fmt="d", cmap="coolwarm")
    plt.tight_layout()
