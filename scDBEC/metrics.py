import numpy as np
import pandas as pd
import scanpy as sc
from harmonypy import compute_lisi
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import pair_confusion_matrix
from sklearn.neighbors import NearestNeighbors

def cal_ilisi(emb, meta):
    lisi_index = compute_lisi(emb, meta, ['celltype','BATCH'])
    ilisi = np.median(lisi_index[:,1])
    return ilisi

def silhouette(adata, group_key, batch_key,embed, metric='euclidean'):
    """
    Silhouette score of batch labels subsetted for each group.
    params:
        batch_key: batches to be compared against
        group_key: group labels to be subsetted by e.g. cell type
        embed: name of column in adata.obsm
        metric: see sklearn silhouette score
    """
    asw_celltype = silhouette_score(
        X=adata.obsm[embed],
        labels=adata.obs[group_key],
        metric=metric
    )

    asw_batch = silhouette_score(
        X=adata.obsm[embed],
        labels=adata.obs[batch_key],
        metric=metric
    )

    min_val = -1
    max_val = 1
    asw_batch_norm = (asw_batch - min_val) / (max_val - min_val)
    asw_celltype_norm = (asw_celltype - min_val) / (max_val - min_val)
    fscoreASW = (2 * (1 - asw_batch_norm)*(asw_celltype_norm))/(1 - asw_batch_norm + asw_celltype_norm)
    return asw_celltype_norm, asw_batch_norm, fscoreASW

def find_resolution(adata_, n_clusters, random):
    adata = adata_.copy()
    obtained_clusters = -1
    iteration = 0
    resolutions = [0., 1000.]
    while obtained_clusters != n_clusters and iteration < 50:
        current_res = sum(resolutions)/2
        sc.tl.louvain(adata, resolution = current_res, random_state = random)
        labels = adata.obs['louvain']
        obtained_clusters = len(np.unique(labels))

        if obtained_clusters < n_clusters:
            resolutions[0] = current_res
        else:
            resolutions[1] = current_res
        iteration = iteration + 1
    return current_res

def ari(labels_true,labels_pred): 
    '''safer implementation of ari score calculation'''
    (tn, fp), (fn, tp) = pair_confusion_matrix(labels_true, labels_pred)
    tn=int(tn)
    tp=int(tp)
    fp=int(fp)
    fn=int(fn)

    # Special cases: empty data or full agreement
    if fn == 0 and fp == 0:
        return 1.0

    return 2. * (tp * tn - fn * fp) / ((tp + fn) * (fn + tn) +
                                       (tp + fp) * (fp + tn))

def calulate_ari_nmi(adata_integrated,n_cluster):
    sc.pp.neighbors(adata_integrated,random_state=0)
    reso=find_resolution(adata_integrated,n_cluster,0)
    sc.tl.louvain(adata_integrated,reso,random_state=0)
    sc.tl.umap(adata_integrated)
    if(adata_integrated.X.shape[1]==2):
        adata_integrated.obsm["X_emb"]=adata_integrated.X
    ARI= ari(adata_integrated.obs["celltype"].astype(str), adata_integrated.obs["louvain"])
    NMI= normalized_mutual_info_score(adata_integrated.obs["celltype"].astype(str), adata_integrated.obs["louvain"])
    print("louvain clustering result(resolution={}):n_cluster={}".format(reso,n_cluster))
    print("ARI:",ARI)
    print("NMI:",NMI)
    return ARI,NMI

def BatchKL(adata, dimensionData=None, replicates=200, n_neighbors=100, n_cells=100, batch="BatchID", embedding_key="X_umap"):
    """
    Compute the KL divergence of batch mixing in a t-SNE plot or other dimensionality-reduced data.
    
    Parameters:
    - adata: AnnData object containing the data.
    - dimensionData: Optional; a matrix or array of dimension-reduced data.
    - replicates: Number of bootstrap iterations.
    - n_neighbors: Number of nearest neighbors to consider for each sampled cell.
    - n_cells: Number of cells to randomly sample in each bootstrap iteration.
    - batch: Column in `adata.obs` containing batch IDs.
    - embedding_key: Key in `adata.obsm` for the dimensionality-reduced data (e.g., t-SNE).
    
    Returns:
    - Mean KL divergence over all bootstrap iterations.
    """
    np.random.seed(42)
    # Extract dimension data
    if dimensionData is None:
        tsnedata = adata.obsm[embedding_key]
    else:
        tsnedata = dimensionData
    
    # Get batch data as a categorical variable
    batchdata = adata.obs[batch].astype('category')
    table_batchdata = batchdata.value_counts().values
    tmp00 = table_batchdata / table_batchdata.sum()  # Proportion of population
    n = adata.n_obs

    KL = []
    for _ in range(replicates):
        bootsamples = np.random.choice(range(n), n_cells, replace=True)
        nbrs = NearestNeighbors(n_neighbors=min(5 * len(tmp00), n_neighbors)).fit(tsnedata)
        distances, indices = nbrs.kneighbors(tsnedata[bootsamples, :])
        KL_x = []
        for i in range(len(bootsamples)):
            ids = indices[i]
            tmp = pd.value_counts(batchdata.iloc[ids]).values
            tmp = tmp / tmp.sum()
            valid_mask = (tmp > 0) & (tmp00 > 0)
            if np.any(valid_mask):
                kl_div = np.sum(tmp[valid_mask] * np.log2(tmp[valid_mask] / tmp00[valid_mask]))
                KL_x.append(kl_div)
        if KL_x:
            KL.append(np.mean(KL_x))
    if KL:
        return np.mean(KL)
    else:
        return np.nan