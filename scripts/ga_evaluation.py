
import sys
from collections import Counter

import hdbscan
import numpy as np
import pandas as pd
import scripts.internal_scores as validation
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.cluster import adjusted_rand_score, silhouette_score, normalized_mutual_info_score
from tqdm import tqdm
import scanpy.api as sc

sys.path.append("..")

def random_sampling(data, truth, n_clusters, algo = "gmm"):
    scores = []
    for i in tqdm(range(100)):
        features = np.random.choice(np.arange(data.shape[1]), replace = False, size = 10)
        input_data = data[:, features]
        if algo == "gmm":
            gmm = mixture.GaussianMixture(n_components=n_clusters,
                              covariance_type="full", random_state=0)
            pred = gmm.fit_predict(input_data)
        else:
            pred = hdbscan.HDBSCAN(min_cluster_size =10).fit(input_data).labels_

        ari = adjusted_rand_score(truth, pred)
        scores.append(ari)
    return np.median(scores)
def run_leiden(data, leiden_n_neighbors=300):
    """
    Performs Leiden community detection on given data.

    Args:
        data ([type]): [description]
        n_neighbors (int, optional): [description]. Defaults to 10.
        n_pcs (int, optional): [description]. Defaults to 40.

    Returns:
        [type]: [description]
    """
    import scanpy.api as sc
    n_pcs = 0
    adata = sc.AnnData(data)
    if leiden_n_neighbors > len(data)//3:
        leiden_n_neighbors = len(data)//3
    sc.pp.neighbors(adata, n_neighbors=leiden_n_neighbors,
                    n_pcs=n_pcs, use_rep='X')
    sc.tl.leiden(adata)
    pred = adata.obs['leiden'].to_list()
    pred = [int(x) for x in pred]
    return pred
def clustering_evaluation(features, data, params, full_eval = False):
    """
    Cluster subspace given by features and compute the unsupervised scores
    If the ground truth is available (y) also compute ARI
    """
    if "multi_n_clusters" in params and params["multi_n_clusters"] is not None: 
        return clustering_evaluation_multi_k(features, data, params)
    
    
    if len(features) >2 and params["pca"] == True:
        input_data  = data[:, features]
        input_data = PCA(2).fit_transform(input_data)
    else:
        input_data  = data[:, features]
        
    if "clustering" not in params or params["clustering"] == "KMeans":
        pred = KMeans(n_clusters=params["n_clusters"],
                      random_state=0).fit(input_data).labels_
    if params["clustering"] == "hdbscan":
        pred = hdbscan.HDBSCAN(min_cluster_size =params["hdbscan_min_cluster_size"]
                              ).fit(data[:, features]).labels_
        
    if params["clustering"] == "gmm":
        gmm = mixture.GaussianMixture(n_components=params["n_clusters"],
                      covariance_type="full", random_state=0)
        pred = gmm.fit_predict(input_data)
        
    if params["clustering"] == "leiden":
        pred = run_leiden(input_data)

    frequencies = np.array(list(Counter(pred).values()))
    nb_invalid_clusters = len(
        frequencies[frequencies < params["min_cluster_size"]])
    scores = {}
    val = validation.validation()
    for i, method in enumerate(params["methods"]):
        if nb_invalid_clusters > 0:
            scores[method] = -2
        else:
            scores[method] = getattr(val,
                                     method)(data[:, features], pred)

    if params.get("y", None) is not None:
        for method in params["truth_methods"]:
            scores[method] = round(getattr(val, method)(params["y"], pred), 2)
            
    if full_eval:
        if params.get("y", None) is not None:
            scores["ari"] = round(getattr(val, "ari")(params["y"], pred), 2)
            scores["nmi"] = round(getattr(val, "nmi")(params["y"], pred), 2)
        scores["silhouette"] = round(getattr(val, "silhouette")(data[:, features], pred), 2)
        scores["adapted_ratkowsky_lance"] = round(getattr(val, "adapted_ratkowsky_lance")(data[:, features], pred), 2)
        scores["point_biserial"] = round(getattr(val, "point_biserial")(data[:, features], pred), 2)
        
    scores['structure'] = f'{Counter(pred)}'
    return scores, pred

def clustering_evaluation_multi_k(features, data, params):
    """
    Cluster subspace given by features and compute the unsupervised scores
    If the ground truth is available (y) also compute ARI
    """
    preds = []
    multi_n_clusters = params["multi_n_clusters"]
    input_data  = data[:, features]
    
    for n_clusters in multi_n_clusters:
        if params["clustering"] == "KMeans":
            pred = KMeans(n_clusters=n_clusters,
                      random_state=0).fit(input_data).labels_
        if params["clustering"] == "gmm":
            gmm = mixture.GaussianMixture(n_components=n_clusters,
                      covariance_type="full", random_state=0)
            pred = gmm.fit_predict(input_data)
        preds.append(pred)
    
    frequencies = np.array(list(Counter(pred).values()))
    nb_invalid_clusters = len(
        frequencies[frequencies < params["min_cluster_size"]])
    scores = {}
    val = validation.validation()
    for i, method in enumerate(params["methods"]):
        scores[method] = []
        for pred in preds:
            frequencies = np.array(list(Counter(pred).values()))
            nb_invalid_clusters = len(
                frequencies[frequencies < params["min_cluster_size"]])
            if nb_invalid_clusters > 0:
                scores[method].append(-2)
            else:
                scores[method].append(getattr(val,
                                         method)(data[:, features], pred))
        sel_pred_idx = np.argmax(scores[method])
#         print(sel_pred_idx, end = ";")
        scores[method] = scores[method][sel_pred_idx]
        
    if params.get("y", None) is not None:
        for method in params["truth_methods"]:
            scores[method] = round(getattr(val, method)(params["y"], preds[sel_pred_idx]), 2)
    scores['structure'] = f'{Counter(preds[sel_pred_idx])}'
    return scores, pred


def evaluateIndividual(features, data, params, archive=None):
    """
    Cluster subspace given by features and compute the unsupervised scores
    If the ground truth is available (y) also compute ARI
    """
    features_string = '_'.join(np.sort(features).astype(str))

    if archive is not None and "features" in archive.columns and features_string in archive["features"].values:
        return archive[archive["features"] == features_string].to_dict(
            "records")[0], archive
    scores = {}

    clustering_scores, _ = clustering_evaluation(features, data, params)
    scores = {**scores, **clustering_scores}

    scores['features'] = features_string
    scores['size'] = len(features)
    # If a 2d subspace better than what we have in the archive has been discovered, add it to the archive
    if params is not None and len(features) == 2 and params.get("archive_2d", None) is not None:
        if (params['loss'] in params["archive_2d"].columns and
            scores[params['loss']] >
                params["archive_2d"][params['loss']].min() and
                features_string not in params["archive_2d"]["features"].values
            ):
            scores_2d = scores.copy()
            scores_2d["feature1"] = features[0]
            scores_2d["feature2"] = features[1]
            params["archive_2d"] = params["archive_2d"].append(
                scores_2d, ignore_index=True,
                sort=False).sort_values(by=params['loss'],
                                        ascending=False)
            params["archive_2d"] = params["archive_2d"].reset_index(drop=True)

    if archive is None:
        archive = pd.DataFrame()
    archive = archive.append(scores, ignore_index=True, sort=False)

    return scores, archive



def evaluate_ga_result(globalResults, best_subspaces, truths):
    
    keys = list(globalResults.keys())
    columns = [
        "experiment", "true_subspace_id", "pred_subspace_id", "%intersect", "iou",
        "extra_features", "true_subspace", "pred_subspace", "missed_features",
        "ari", "nmi", "true_nb_clust", "pred_nb_clust"
    ]
    eval_df = pd.DataFrame(columns=columns)
    for experiment in keys:
        df = globalResults[experiment]
        features = df['features'].values
        predicted_partitions = df['partition'].values
        for sid, subspace in enumerate(best_subspaces):
            subspace_df = pd.DataFrame(columns=columns)
            for i in range(len(features)):
                cur_subspace = features[i]
                intersect_size = len(np.intersect1d(cur_subspace, subspace)) 
                union_size = len(np.unique(np.concatenate([cur_subspace, subspace])))
                iou = intersect_size/union_size
                identified_features = intersect_size/ len(subspace)
                
                extra_features = len(np.setdiff1d(cur_subspace, subspace))
   
                missed_features = np.setdiff1d(subspace, cur_subspace)
                ari = adjusted_rand_score(truths[sid], predicted_partitions[i])
                nmi = normalized_mutual_info_score(truths[sid], predicted_partitions[i])
                
                nb_clust_truth = len(np.unique(truths[sid]))
                nb_clust_pred = len(np.unique(predicted_partitions[i]))
                subspace_df.loc[subspace_df.shape[0]] = [
                    experiment, sid, i, identified_features, iou, extra_features,
                    subspace, cur_subspace, missed_features, ari, nmi, nb_clust_truth,
                    nb_clust_pred
                ]
            subspace_df = subspace_df[subspace_df["%intersect"] ==
                                      subspace_df["%intersect"].max()]
            if subspace_df["%intersect"].max() == 0:
                subspace_df = subspace_df.iloc[:1]
            eval_df = pd.concat([eval_df, subspace_df],
                                ignore_index=True,
                                sort=False)
    return eval_df
