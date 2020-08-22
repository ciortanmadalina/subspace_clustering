import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from sklearn import preprocessing


def preprocess_rna(df,
                   meta,
                   truth_column,
                   truth_values,
                   filename,
                   metric='euclidean',
                   normalize=True,
                   path='data/rna_data'):
    """[summary]

    Arguments:
        df {[type]} -- [description]
        meta {[type]} -- [description]
        truth_column {[type]} -- [description]
        truth_values {[type]} -- [description]
        filename {[type]} -- [description]

    Keyword Arguments:
        metric {str} -- [description] (default: {'euclidean'})
        normalize {bool} -- [description] (default: {True})
        path {str} -- [description] (default: {'data/rna_data'})
    """

    if normalize == False:
        filename = f"{filename}_no_norm"
    zname = "Z"
    if metric != 'euclidean':
        zname = f"{zname}_{metric}"
    y = meta[meta["bcr_patient_barcode"] == truth_column].values[0][1:]
    ids = meta.columns[1:].to_list()

    meta_id_to_y = {ids[i]: y[i] for i in range(len(y))}
    patient_labels = df.columns[1:]
    map_id_label = {}

    for i in range(len(ids)):
        for p in patient_labels:
            if ids[i].upper() in p:
                map_id_label[p] = meta_id_to_y[ids[i]]
                break

    patients = list(map_id_label.keys())
    patients_class = list(map_id_label.values())
    df = df.drop(0)
    genes = df["Hybridization REF"].values
    df = df[patients].T
    df.columns = genes

    df["y"] = patients_class
    df = df[df["y"].isin(truth_values)]
    df["y"] = preprocessing.LabelEncoder().fit_transform(df["y"].values)

    truth = df["y"].values
    df = df.drop("y", axis=1)
    df = df.astype(float)
    df = df.loc[:, (df != df.iloc[0]).any()]  # remove constant columns
    df = df.dropna(axis=1)
    df = df.apply(np.log1p)
    # remove features where more than 75% of values are under 15% threshold
    percentile_threshold = np.percentile(np.concatenate(df.values), 15)
    expression = ((df < percentile_threshold).astype(int).sum(axis=0) /
                  df.shape[0]).values

    genes_to_keep = np.where(expression < 0.75)[0]
    df = df[df.columns[genes_to_keep]]

    ids = df.index.values
    genes = df.columns

    # normalize data
    if normalize:
        df = preprocessing.MinMaxScaler().fit_transform(df)
        df = pd.DataFrame(df)

    # Save hierarchical clustering
    Z = linkage(df.T, method='complete', metric=metric)
    z_file = f"{path}/{filename}_{zname}.npy"
    np.save(z_file, Z)

    df.columns = genes
    df["y"] = truth
    df = df.set_index(ids)
    df.to_pickle(f"{path}/{filename}.pkl")

    meta_ids = meta.columns[1:].to_list()
    patient_labels = df.index.values
    map_id_label = {}
    for i in range(len(meta_ids)):
        for p in patient_labels:
            if meta_ids[i].upper() in p:
                map_id_label[meta_ids[i]] = p
                break
    meta = meta.T
    columns = meta.values[0, :]

    meta.columns = columns
    meta.drop(["bcr_patient_uuid", "patient_id"], axis=1, inplace=True)
    meta = meta.reset_index()
    meta["id"] = meta["index"].apply(lambda x: map_id_label.get(x, np.nan))
    additional_df = pd.merge(df[["y"]], meta, left_index=True,
                             right_on="id").drop(["index", "y"], axis=1)
    additional_df = additional_df.reset_index(drop=True)
    additional_df.to_pickle(f"{path}/{filename}_additional.pkl")
