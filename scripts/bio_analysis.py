import itertools
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scripts.internal_scores as validation
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.manifold import TSNE
sys.path.append("..")


def solutions_to_survival_analysis(additional_df, solutions, filename):
    """[summary]

    Arguments:
        additional_df {[type]} -- [description]
        solutions {[type]} -- [description]
        filename {[type]} -- [description]
    """
    additional_df = additional_df.copy()
    predictions = solutions["partition"].values
    for i in range(len(predictions)):
        additional_df[f"subspace_{i}"] = predictions[i]
    additional_df.to_pickle(filename)


def clinical_data_analysis(additional_df, solutions, n_clusters):
    """[summary]

    Arguments:
        additional_df {[type]} -- [description]
        solutions {[type]} -- [description]
        n_clusters {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    subspace_pred = solutions["partition"]
    all_classes_raw = {}
    all_classes = {}
    for i, c in enumerate(additional_df.columns):
        try:
            if additional_df[c].dropna().unique().shape[0] <= 1:
                print(f"No more than 1 class found for {c}")
                continue
            all_classes_raw[c] = additional_df[c].values
            not_null_idx = additional_df[~additional_df[c].isnull(
            )].index.values
            null_idx = additional_df[additional_df[c].isnull()].index.values

            values = additional_df[c].astype(float).values
            if len(np.unique(values)) < n_clusters:
                all_classes[c] = values
            else:
                print(f"Clustering numeric values for {c}")
                if len(not_null_idx) > n_clusters * 2:
                    predK = KMeans(n_clusters=n_clusters, random_state=0).fit(
                        values[not_null_idx].reshape(-1, 1)).labels_
                    prediction = np.zeros(additional_df.shape[0])
                    prediction[not_null_idx] = predK
                    prediction[null_idx] = np.nan
                    all_classes[c] = prediction
        except ValueError as e:
            n_unique = additional_df[c].unique().shape[0]
            if n_unique > len(subspace_pred[0]) // 3:  # too many categories
                print(f"Found {n_unique} values for {c}, skipping")
            else:
                labels = preprocessing.LabelEncoder().fit_transform(
                    additional_df[c].values[not_null_idx])
                all_classes[c] = labels
                prediction = np.zeros(additional_df.shape[0])
                prediction[not_null_idx] = labels
                prediction[null_idx] = np.nan
                all_classes[c] = prediction
                print(f"Found {n_unique} values for {c}")

    additional_results = pd.DataFrame(
        columns=["subspace", "additional_data", "ari", "n"])
    for i, prediction in enumerate(subspace_pred):
        for name, values in all_classes.items():
            idx = np.where(~np.isnan(all_classes[name]))[0]
            ari = round(
                adjusted_rand_score(prediction[idx], all_classes[name][idx]),
                2)
            additional_results.loc[additional_results.shape[0]] = [
                i, name, ari, len(idx)
            ]

    best_subspace_match = pd.merge(
        additional_results.groupby("subspace")[["ari"]].max().reset_index(),
        additional_results,
        left_on=["subspace", "ari"],
        right_on=["subspace", "ari"])
    best_meta_subspace = pd.merge(
        additional_results.groupby("additional_data")[["ari"
                                                       ]].max().reset_index(),
        additional_results,
        left_on=["additional_data", "ari"],
        right_on=["additional_data", "ari"])
    return additional_results, best_subspace_match, best_meta_subspace


def plot_pca_subspaces(data, solutions, method, truth, pdf_name=None, name=""):
    """[summary]

    Arguments:
        data {[type]} -- [description]
        solutions {[type]} -- [description]
        method {[type]} -- [description]
        truth {[type]} -- [description]

    Keyword Arguments:
        pdf_name {[type]} -- [description] (default: {None})
        name {str} -- [description] (default: {""})
    """
    if pdf_name is not None:
        pdf = PdfPages(pdf_name)
    else:
        pdf = None
    subspaces = solutions["features"].values
    partitions = solutions["partition"].values
    scores = solutions[method].values
    for k, subspace in enumerate(subspaces):

        pca = PCA(2)
        data_x = data[:, subspace]
        pca_data = pca.fit_transform(data_x)
        predK = partitions[k]
        ari = adjusted_rand_score(truth, predK)

        score = scores[k]
        plt.figure(figsize=(10, 3))
        plt.subplot(121)
        plt.title(
            f"Prediction: {name} {method} {round(score,2)} \nfor subspace of {len(subspace)} feaures"
        )
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=predK)

        plt.subplot(122)
        if truth is not None:
            plt.title(f"{k+1}) Truth: {name} PCA Truth, ari = {round(ari,2)}")
            plt.scatter(pca_data[:, 0], pca_data[:, 1], c=truth)
        plt.tight_layout()
        if pdf is not None:
            pdf.savefig()
            plt.close('all')
        else:
            plt.show()

    if pdf is not None:
        pdf.close()


def merge_subspaces(solutions, method, data, n_clusters, truth):
    """[summary]

    Arguments:
        solutions {[type]} -- [description]
        method {[type]} -- [description]
        data {[type]} -- [description]
        n_clusters {[type]} -- [description]
        truth {[type]} -- [description]

    Returns:
        [type] -- [description]
    """

    prediction_scores = solutions[method].values
    predictions = solutions["partition"].values
    subspaces = solutions["features"].values
    merged = pd.DataFrame(columns=[
        "s1", "s2", "s1_s2_ari", "ari", "score", "score_1", "score_2",
        "better_s1", "better_s2"
    ])
    for i, j in list(itertools.combinations(np.arange(len(predictions)), 2)):
        subspace = np.concatenate([subspaces[i], subspaces[j]])
        data_x = data[:, subspace]
        predK = KMeans(n_clusters=n_clusters,
                       random_state=0).fit(data_x).labels_
        ari = adjusted_rand_score(truth, predK)
        val = validation.validation()
        score = getattr(val, method)(data_x, predK)
        previous_max = max(prediction_scores[i], prediction_scores[j])
        ari_s1_s2 = adjusted_rand_score(predictions[i], predictions[j])
        merged.loc[merged.shape[0]] = [
            i, j, ari_s1_s2, ari, score, prediction_scores[i],
            prediction_scores[j], score > prediction_scores[i],
            score > prediction_scores[j]
        ]
        merged = merged.sort_values(by="s1_s2_ari", ascending=False)
    return merged

def plot_prediction_vs_ground_truth(k, solutions, data, truth, do_pca = True, filename = None):
    predK = solutions["partition"].values[k]
    subspace = solutions["features"].values[k]
    ari = solutions["ari"].values[k]
    score = round(solutions[solutions.columns[0]].values[k],2)
    data_x = data[:, subspace]
    if do_pca:
        pca_data = PCA(n_components=2).fit_transform(data_x)
    else:
        pca_data = TSNE(n_components=2).fit_transform(data_x)

    plt.figure(figsize=(10, 3))
    ax = plt.subplot(121)
    plt.title(
        f"(a) Prediction result on discovered subspace ({len(subspace)} feaures)\n" +
        f"Internal score {score} with ARI {ari}"
    )
    plt.scatter(pca_data[:, 0], pca_data[:, 1], c=predK, cmap = "coolwarm", alpha = 0.6, s = 8)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    ax = plt.subplot(122)
    if truth is not None:
        method = "PCA" if do_pca else "TSNE"
        plt.title(f"(b) Ground truth annotation on the same subspace\n Using {method}")
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=truth, alpha = 0.6, s = 8)
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticklabels([])
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')

def plot_subspaces_with_best_meta(solutions, best_subspace_match, data, 
                                 ground_truth_nb, maping,do_pca = True, filename = None):

    subspaces = solutions["features"].values
    partitions = solutions["partition"].values
    scores = solutions[solutions.columns[0]].values
    ncols = 5
    nrows = solutions.shape[0]//ncols
    plt.figure(figsize = (16, nrows * 3) )
    for k, subspace in enumerate(subspaces):

        data_x = data[:, subspace]
        if len(subspace) ==2:
            pca_data= data[:, subspace]
        else:
            if do_pca:
                pca_data = PCA(n_components=2).fit_transform(data_x)
            else:
                pca_data = TSNE(n_components=2).fit_transform(data_x)
        predK = partitions[k]
#         predK = KMeans(n_clusters=n_clusters,
#                            random_state=0).fit(data_x).labels_
        title = maping[best_subspace_match["additional_data" ].values[k]]

        score = best_subspace_match['ari'].values[k]
        ax= plt.subplot(nrows, ncols, k+1)
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
        fontweight = "normal" if k!= ground_truth_nb else "bold"
        plt.title(
            f"{k+1}) {title}\n ARI {score}  ({len(subspace)} feaures)",
            fontweight = fontweight
        )
        plt.scatter(pca_data[:, 0], pca_data[:, 1], c=predK, cmap = "coolwarm", alpha = 0.4, s = 6)
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
