# scDBEC
### Overview  

**scDBEC** (single-cell Deep learning Batch Effect Correction) is an unsupervised algorithm designed to address batch effects in single-cell RNA sequencing (scRNA-seq) data while simultaneously clustering cells. By leveraging inter-batch information, scDBEC integrates data from multiple batches into a unified network without requiring additional annotations.  

The model utilizes an ZINB model -based autoencoder guided by three integrated loss functions:  
- **Reconstruction Loss**: Ensures accurate reconstruction of gene expression profiles, preserving biological characteristics.  
- **Clustering Loss**: Captures local relationships by minimizing divergence between high- and low-dimensional affinity matrices.  
- **MMD Loss**: Aligns latent space representations across batches, correcting for batch effects effectively.  

For a detailed description of the implementation and to reproduce the results presented in our manuscript, please visit the GitHub repository: [https://github.com/XinhanSONG/scDBEC](#).

### Dependency & Installation 
The programs require a PyTorch environment and the following dependencies to be installed:  

- **Python**: 3.8.19  
- **PyTorch**: 2.3.1  
- **NumPy**: 1.22.0  
- **Pandas**: 2.0.3  
- **Scikit-learn**: 1.3.2  
- **Matplotlib**: 3.7.5  
- **Scanpy**: 1.9.8  
- **Anndata**: 0.9.2  
- **Harmonypy**: 0.0.9  
- **Natsort**: 8.4.0  

Ensure you have Python 3.8.19 installed. Create and activate a virtual environment. Follow the steps below to set up the environment and install the required dependencies for **scDBEC**:  

```bash  
git clone https://github.com/username/scDBEC  
python3 -m venv scdbec_env  
pip install -r requirements.txt  
# Check that all packages are installed correctly:  
python -m pip list  
```  
You are now ready to run **scDBEC**!