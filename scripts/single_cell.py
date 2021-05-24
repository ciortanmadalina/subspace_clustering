import scanpy.api as sc
import scipy as sp
import numpy as np
import pandas as pd
def filter_data(X,  highly_genes=500):
    """
    Remove less variable genes

    Args:
        X ([type]): [description]
        highly_genes (int, optional): [description]. Defaults to 500.

    Returns:
        [type]: [description]
    """

    X = np.ceil(X).astype(np.int)
    adata = sc.AnnData(X)

    sc.pp.filter_genes(adata, min_counts=3)
    sc.pp.filter_cells(adata, min_counts=1)
    sc.pp.normalize_per_cell(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=4,
                                min_disp=0.5, n_top_genes=highly_genes, subset=True)
    genes_idx = np.array(adata.var_names.tolist()).astype(int)
    cells_idx = np.array(adata.obs_names.tolist()).astype(int)

    return genes_idx, cells_idx

def normalize(adata, copy=True, highly_genes = None, filter_min_counts=True, 
              size_factors=True, normalize_input=True, logtrans_input=True):
    """
    Normalizes input data and retains only most variable genes 
    (indicated by highly_genes parameter)

    Args:
        adata ([type]): [description]
        copy (bool, optional): [description]. Defaults to True.
        highly_genes ([type], optional): [description]. Defaults to None.
        filter_min_counts (bool, optional): [description]. Defaults to True.
        size_factors (bool, optional): [description]. Defaults to True.
        normalize_input (bool, optional): [description]. Defaults to True.
        logtrans_input (bool, optional): [description]. Defaults to True.

    Raises:
        NotImplementedError: [description]

    Returns:
        [type]: [description]
    """
    if isinstance(adata, sc.AnnData):
        if copy:
            adata = adata.copy()
    elif isinstance(adata, str):
        adata = sc.read(adata)
    else:
        raise NotImplementedError
    norm_error = 'Make sure that the dataset (adata.X) contains unnormalized count data.'
    assert 'n_count' not in adata.obs, norm_error
    if adata.X.size < 50e6: # check if adata.X is integer only if array is small
        if sp.sparse.issparse(adata.X):
            assert (adata.X.astype(int) != adata.X).nnz == 0, norm_error
        else:
            assert np.all(adata.X.astype(int) == adata.X), norm_error

    if filter_min_counts:
        sc.pp.filter_genes(adata, min_counts=1)#3
        sc.pp.filter_cells(adata, min_counts=1)
    if size_factors or normalize_input or logtrans_input:
        adata.raw = adata.copy()
    else:
        adata.raw = adata
    if size_factors:
        sc.pp.normalize_per_cell(adata)
        adata.obs['size_factors'] = adata.obs.n_counts / np.median(adata.obs.n_counts)
    else:
        adata.obs['size_factors'] = 1.0
    if logtrans_input:
        sc.pp.log1p(adata)
    if highly_genes != None:
        sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5, n_top_genes = highly_genes, subset=True)
    if normalize_input:
        sc.pp.scale(adata)
    return adata

def preprocess_data(X, nb_genes = 500, normalize_input=False):
    """
    Preprocessing phase as proposed in scanpy package.
    Keeps only nb_genes most variable genes and normalizes
    the data to 0 mean and 1 std.
    Args:
        X ([type]): [description]
        nb_genes (int, optional): [description]. Defaults to 500.
    Returns:
        [type]: [description]
    """
    X = np.ceil(X).astype(np.int)
    count_X = X
    print(X.shape, count_X.shape, f"keeping {nb_genes} genes")
    orig_X = X.copy()
    adata = sc.AnnData(X)

    adata = normalize(adata,
                      copy=True,
                      highly_genes=nb_genes,
                      size_factors=True,
                      normalize_input=normalize_input,
                      logtrans_input=True)
    X = adata.X.astype(np.float32)
    return X