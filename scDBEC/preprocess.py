import numpy as np
import scanpy as sc 

def preprocess(adata, target_sum=1e4, n_top_genes=1000, max_value=1.0):
    """
    Preprocesses the input AnnData object.

    Args:
        adata (anndata.AnnData): The input AnnData object.
        target_sum (float, optional): Target total counts after normalization. Defaults to 1e4.
        n_top_genes (int, optional): Number of highly variable genes to select. Defaults to 1000.
        max_value (float, optional): Maximum value after scaling. Defaults to 1.0.

    Returns:
        anndata.AnnData: Preprocessed AnnData object.
    """
    # Normalize total counts
    sc.pp.normalize_total(adata, target_sum=target_sum)
    
    # Logarithmize data
    sc.pp.log1p(adata)
    
    # Identify highly variable genes
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes, subset=True)
    
    # Scale data based on batches
    adata_sep = []
    for batch in np.unique(adata.obs["BATCH"]):
        sep_batch = adata[adata.obs["BATCH"] == batch].copy()
        sc.pp.scale(sep_batch, max_value=max_value)
        adata_sep.append(sep_batch)
    
    # Concatenate processed batches
    adata_processed = sc.AnnData.concatenate(*adata_sep)
    
    return adata_processed