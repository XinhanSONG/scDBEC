from sklearn.cluster import KMeans
from scDBEC.preprocess import load_data
import numpy as np
from anndata import AnnData
import scanpy as sc 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold._t_sne import _joint_probabilities

def getdims(x=(10000,200)):
    """
    This function will give the suggested nodes for each encoder layer
    return the dims for network
    """
    assert len(x)==2
    n_sample=x[0]
    if n_sample>25000:# may be need complex network
        dims=[x[-1],256,128]
    elif n_sample>10000:#10000
        dims=[x[-1],128,32]
    elif n_sample>5000: #5000
        dims=[x[-1],32,16] #16
    elif n_sample>2000:
        dims=[x[-1],128]
    elif n_sample>500:
        dims=[x[-1],64]
    else:
        dims=[x[-1],16]
    return dims

def generate_P(X, device, perplexity=30):
    """
    Generate the probability distribution matrix P.
    :param X: Input data (torch.Tensor)
    :param device: Device (str, e.g., 'cpu' or 'cuda')
    :param perplexity: Desired perplexity (int)
    :return: Probability distribution matrix P (torch.Tensor)
    """
    distances = pairwise_distances(X.detach().cpu().numpy(), metric='euclidean', squared=True)
    P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)
    P_hat = squareform(P)
    return torch.tensor(P_hat, device=device)

def generate_Q(Z, device):
    """
    Generate the probability distribution matrix Q.
    :param Z: Embedded representations (torch.Tensor)
    :param device: Device (str, e.g., 'cpu' or 'cuda')
    :return: Probability distribution matrix Q (torch.Tensor)
    """
    Z = Z.detach().cpu().numpy()
    distance_matrix = pairwise_distances(Z, metric='euclidean', squared=True)
    inverse = 1 / (distance_matrix + 1)
    np.fill_diagonal(inverse, 0)  # Set diagonal elements to 0 more concisely
    inverse_sum = np.sum(inverse)  # Simplify normalization calculation
    Q_normalized = inverse / inverse_sum
    return torch.tensor(Q_normalized, device=device)

class KLLoss(torch.nn.Module):
    """
    Compute the Kullback-Leibler (KL) divergence between two probability distributions.
    """
    def __init__(self):
        super(KLLoss, self).__init__()

    def forward(self, P: torch.Tensor, Q: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence between two probability distributions.

        :param P: Pairwise similarity matrix for high-dimensional space (torch.Tensor) of shape (n, n).
        :param Q: Pairwise similarity matrix for low-dimensional space (torch.Tensor) of shape (n, n).
        :return: KL divergence (torch.Tensor) as a scalar value.
        """
        # Avoid log(0) by adding a small epsilon
        epsilon = 1e-12
        P = torch.clamp(P, epsilon, 1.0)
        Q = torch.clamp(Q, epsilon, 1.0)
        # Compute element-wise KL divergence
        kl_divergence = torch.sum(P * torch.log(P / Q))
        return kl_divergence

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MeanAct(nn.Module):
    def __init__(self):
        super(MeanAct, self).__init__()

    def forward(self, x):
        return torch.clamp(torch.exp(x), min=1e-5, max=1e6)

class DispAct(nn.Module):
    def __init__(self):
        super(DispAct, self).__init__()

    def forward(self, x):
        return torch.clamp(F.softplus(x), min=1e-4, max=1e4)

class ZINBLoss(nn.Module):
    def __init__(self):
        super(ZINBLoss, self).__init__()

    def forward(self, x, mean, disp, pi, scale_factor=1.0, ridge_lambda=0.0):
        eps = 1e-10
        #scale_factor = scale_factor[:, None]
        mean = mean * scale_factor
        
        t1 = torch.lgamma(disp+eps) + torch.lgamma(x+1.0) - torch.lgamma(x+disp+eps)
        t2 = (disp+x) * torch.log(1.0 + (mean/(disp+eps))) + (x * (torch.log(disp+eps) - torch.log(mean+eps)))
        nb_final = t1 + t2

        nb_case = nb_final - torch.log(1.0-pi+eps)
        zero_nb = torch.pow(disp/(disp+mean+eps), disp)
        zero_case = -torch.log(pi + ((1.0-pi)*zero_nb)+eps)
        result = torch.where(torch.le(x, 1e-8), zero_case, nb_case)
        
        if ridge_lambda > 0:
            ridge = ridge_lambda*torch.square(pi)
            result += ridge
        
        result = torch.mean(result)
        return result

class MMDLoss(nn.Module):
    def __init__(self, kernel_type='rbf', kernel_mul=2.0, kernel_num=5, fix_sigma=1, **kwargs):
        super(MMDLoss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma
        self.kernel_type = kernel_type

    def guassian_kernel(self, source, target, kernel_mul, kernel_num, fix_sigma):
        n_samples = int(source.size()[0]) + int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(
            int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i)
                          for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp)
                      for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def linear_mmd2(self, f_of_X, f_of_Y):
        loss = 0.0
        delta = f_of_X.float().mean(0) - f_of_Y.float().mean(0)
        loss = delta.dot(delta.T)
        return loss

    def forward(self, source, target):
        if self.kernel_type == 'linear':
            return self.linear_mmd2(source, target)
        elif self.kernel_type == 'rbf':
            batch_size = int(source.size()[0])
            kernels = self.guassian_kernel(
                source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
            XX = torch.mean(kernels[:batch_size, :batch_size])
            YY = torch.mean(kernels[batch_size:, batch_size:])
            XY = torch.mean(kernels[:batch_size, batch_size:])
            YX = torch.mean(kernels[batch_size:, :batch_size])
            loss = torch.mean(XX + YY - XY - YX)
            return loss

class scDBEC(nn.Module):
    def __init__(self, adata, device = 'cuda', batch_size = 500,random_seed = 42,fix_sigma = 1,):
        super(scDBEC, self).__init__()
        dims = getdims(adata.shape)
        input_dim = dims[0] 
        latent_dim = dims[-1]  
        hidden_dims = dims[1:-1]
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            #encoder_layers.append(nn.BatchNorm1d(h_dim)) 
            encoder_layers.append(nn.ReLU())
            #encoder_layers.append(nn.Dropout(p=0.5))
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self._enc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            #decoder_layers.append(nn.BatchNorm1d(h_dim)) 
            decoder_layers.append(nn.ReLU())
            #decoder_layers.append(nn.Dropout(p=0.5))
            in_dim = h_dim
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Output layers
        self._dec_mean = nn.Sequential(
            nn.Linear(hidden_dims[0], input_dim),
            MeanAct()
        )
        self._dec_disp = nn.Sequential(
            nn.Linear(hidden_dims[0], input_dim),
            DispAct()
        )
        self._dec_pi = nn.Sequential(
            nn.Linear(hidden_dims[0], input_dim),
            nn.Sigmoid()
        )
        
        self.device = device
        self.dataloaders = load_data(adata, batch_size=batch_size, device=device, seed = random_seed)
        # self.mu = torch.Tensor(n_clusters, latent_dim).to(self.device)
        self.kl_loss = KLLoss()
        self.zinb_loss = ZINBLoss()
        self.mmd_loss = MMDLoss(fix_sigma=fix_sigma)

    # def soft_assign(self, z):
    #     q = 1.0 / (1.0 + torch.sum((z.unsqueeze(1) - self.mu)**2, dim=2) / self.alpha)
    #     q = q**((self.alpha + 1.0) / 2.0)
    #     q = (q.t() / torch.sum(q, dim=1)).t()
    #     return q
    
    # def target_distribution(self, q):
    #     p = q**2 / (q.sum(0)+ 1e-10)
    #     return (p.t() / (p.sum(1)+1e-10)).t()
    
    def forward(self, x):
        h = self.encoder(x)
        z = self._enc_mu(h)
        h_dec = self.decoder(z)
        #q = self.soft_assign(z)  
        mean = self._dec_mean(h_dec)
        disp = self._dec_disp(h_dec)
        pi = self._dec_pi(h_dec)
        return z, mean, disp, pi
    
    # def kl_loss(self, p, q):
    #     p = torch.clamp(p, min=1e-10)
    #     q = torch.clamp(q, min=1e-10)
    #     return torch.mean(torch.sum(p * torch.log(p / q), dim=-1))
  
    def cycle(self, dataloader):
        while True:
            for batch in dataloader:
                yield batch

    def fit(self, adata, lambda1, lambda2, lr=1., num_epochs=10, weight_decay=1e-2):
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            self.cuda()
        
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=lr, weight_decay=weight_decay)

        cyclers = {batch_name: self.cycle(dl) for batch_name, dl in self.dataloaders.items()}

        # # KMeans Initialization
        # kmeans = KMeans(self.n_cluster, n_init=20)
        # data = self._enc_mu(self.encoder(torch.tensor(adata.X, dtype=torch.float32).cuda()))
        # kmeans.fit(data.cpu().detach().numpy())
        # self.mu.data.copy_(torch.tensor(kmeans.cluster_centers_, device=self.mu.device))

        self.train()
        for epoch in range(num_epochs):
            total_samples = 0
            recon_loss_val = 0
            mmd_loss_val = 0
            kl_loss_val = 0

            start_idxs = {batch_name: 0 for batch_name in self.dataloaders}
            max_len = max(len(dl) for dl in self.dataloaders.values())

            for batch_idx in range(max_len):
                inputs_list = []
                z_list = []
                #q_list = []
                kl_loss_list = []
                recon_loss_list = []
                
                for batch_name, cycler in cyclers.items():
                    inputs = next(cycler).to(self.device)
                    inputs.requires_grad_(True)
                    z, mean, disp, pi = self.forward(inputs)
                    
                    # p_sub = self.target_distribution(q).data[start_idxs[batch_name]:start_idxs[batch_name] + q.shape[0]]
                    # start_idxs[batch_name] = 0 if start_idxs[batch_name] + q.shape[0] == p_sub.shape[0] else start_idxs[batch_name] + q.shape[0]
                    # Compute adjacency matrix P and KL loss
                    P = generate_P(inputs, device=self.device)
                    Q = generate_Q(z, device=self.device)
                    kl_loss = self.kl_loss(P, Q)
                    
                    recon_loss = self.zinb_loss(inputs, mean, disp, pi)
                    
                    inputs_list.append(inputs)
                    z_list.append(z)
                    #q_list.append(q)
                    kl_loss_list.append(kl_loss)
                    recon_loss_list.append(recon_loss)
                
                recon_loss = sum(recon_loss_list)

                # kl_loss = sum(self.kl_loss(self.target_distribution(q), q) for q in q_list)
                kl_loss = sum(kl_loss_list)
                mmd_loss = sum(self.mmd_loss(z_list[i], z_list[j]) for i in range(len(z_list)) for j in range(i + 1, len(z_list)))
                total_loss = recon_loss + lambda1*mmd_loss + lambda2*kl_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

                recon_loss_val += recon_loss.data.item() * len(inputs_list[0])
                mmd_loss_val += mmd_loss.data.item() * len(inputs_list[0])
                kl_loss_val += kl_loss.data.item() * len(inputs_list[0])
                total_samples += len(inputs_list[0])
            logging.info(f"End of Epoch {epoch}: Avg Recon Loss: {recon_loss_val / total_samples:.4f}, Avg MMD Loss: {lambda1*mmd_loss_val / total_samples:.4f}, Avg KL Loss: {lambda2*kl_loss_val / total_samples:.4f}")

    def latent_output(self, adata):
        batches = {}
        for batch_label in np.unique(adata.obs.BATCH):
            batches[batch_label] = adata[adata.obs.BATCH == batch_label]
        z_values = {}

        for batch_label, batch_data in batches.items():
            batch_tensor = torch.tensor(batch_data.X, dtype=torch.float32).to(self.device)
            z, _, _, _ = self.forward(batch_tensor)
            z_values[batch_label] = z.cpu().detach().numpy()

        all_z = np.concatenate([z for z in z_values.values()], axis=0)
        X_umap = AnnData(all_z)
        
        all_batches = sc.AnnData.concatenate(*batches.values())
        X_umap.obs["BATCH"] = list(all_batches.obs.BATCH)
        X_umap.obs["celltype"] = list(all_batches.obs.celltype)

        return X_umap