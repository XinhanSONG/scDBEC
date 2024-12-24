import os
from scDBEC.scDBEC import scDBEC
from scDBEC.preprocess import preprocess
import scanpy as sc
import torch
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from scDBEC.metrics import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load data
data_address = '/root/autodl-tmp/desc-nopage/scDML_main/macaque_raw.h5ad'
adata_raw = sc.read(data_address)
adata = preprocess(adata_raw)

# Define fix_sigma values and parameters
fix_sigma_values = [0.1, 0.5, 1, 2, 5]
total_epochs = 50
pause_interval = 3

# Initialize results storage
results = []
results_dir = "results"
os.makedirs(results_dir, exist_ok=True)

# Training and evaluation loop
for fix_sigma in fix_sigma_values:
    print(f"\n=== Training with fix_sigma = {fix_sigma} ===")
    model = scDBEC(adata=adata, fix_sigma=fix_sigma)
    
    with tqdm(total=total_epochs, desc=f"fix_sigma={fix_sigma}") as pbar:
        for epoch in range(1, total_epochs + 1):
            model.fit(adata=adata, lambda1=1, lambda2=0.1, lr=0.01, num_epochs=1, weight_decay=1e-4)

            # Pause every `pause_interval` epochs for evaluation
            if epoch % pause_interval == 0:
                print(f"Evaluating at epoch {epoch}...")
                X_umap = model.latent_output(adata)
                sc.pp.neighbors(X_umap, use_rep='X', random_state=42)
                sc.tl.umap(X_umap, random_state=42)
                
                # UMAP visualization
                plt.figure()
                sc.pl.umap(X_umap, color=["celltype"], title=f'scDBEC (fix_sigma={fix_sigma}, epoch={epoch})', 
                           show=False, legend_loc='none')
                plt.savefig(os.path.join(results_dir, f"umap_fix_sigma_{fix_sigma}_epoch_{epoch}.png"))
                plt.close()

                # Compute ARI and NMI
                ARI, NMI = calulate_ari_nmi(X_umap, n_cluster=12)
                print(f"Epoch {epoch}, ARI: {ARI:.4f}, NMI: {NMI:.4f}")

                # Save results
                results.append({"fix_sigma": fix_sigma, "epoch": epoch, "ARI": ARI, "NMI": NMI})

            pbar.update(1)

# Save results to a CSV file
results_df = pd.DataFrame(results)
results_path = os.path.join(results_dir, "ari_nmi_results.csv")
results_df.to_csv(results_path, index=False)
print(f"Results saved to {results_path}")
