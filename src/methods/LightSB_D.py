from math import log
import torch.nn as nn
import torch

class LightSB_D(nn.Module):
    def __init__(self, dim, n_cat, n_potentials, beta=1e-5, device='cuda', distr_init='randn', transition='uniform'):
        super().__init__()
        self.dim = dim
        self.n_cat = n_cat
        self.n_potentials = n_potentials
        self.device = device
        
        self.log_alpha = nn.Parameter(torch.zeros(n_potentials, device=device))
        self.log_cp_cores = nn.ParameterList([
            nn.Parameter(torch.zeros(n_potentials, n_cat, device=device))
            for _ in range(dim)
        ])

        self.transition = transition
        self.beta  = beta
        self._initialize_parameters(distr_init)
        
        if transition == 'uniform':
            #According to https://arxiv.org/pdf/2107.03006
            self.log_diff  = log(beta/n_cat + 1e-15) #log((1 - self.beta)/(self.n_cat - 1) + 1e-12)
            self.log_equal = log(1 - ((n_cat - 1)/n_cat)*beta + 1e-15)#log(self.beta + (1 - self.beta)/(self.n_cat - 1) + 1e-12)
        
        if transition == 'gaussian':
            sum_aux = -4 * torch.arange(-self.n_cat + 1, self.n_cat, device=self.device).float()**2 / (self.n_cat - 1)**2
            self.log_Z = torch.logsumexp(sum_aux, dim=0)

    def _initialize_parameters(self, distr_init):
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)
        for core in self.log_cp_cores:
            if distr_init == 'randn':
                nn.init.normal_(core, mean=-1.0, std=0.5)
            elif distr_init == 'uniform':
                nn.init.constant_(core, -torch.log(torch.tensor(self.n_cat * 1.0)))
            else:
                raise ValueError(f"Invalid distr_init: {distr_init}")
            
    def get_log_v(self, y):
        batch_size = y.shape[0]
        log_terms = self.log_alpha[None, :]  # (1, K)
        
        for d in range(self.dim):
            y_d = y[:, d]  # (batch_size,)
            log_r_d = self.log_cp_cores[d][:, y_d].T  # (batch_size, K)
            log_terms = log_terms + log_r_d
            
        log_v = torch.logsumexp(log_terms, dim=1)  # (batch_size,)
        return log_v

    def get_log_c(self, x):
        batch_size = x.shape[0]
        log_z = torch.zeros(batch_size, self.n_potentials, device=self.device)
        
        for d in range(self.dim):
            x_d        = x[:, d]

            if self.transition == 'uniform':
                log_pi_ref = torch.full((batch_size, self.n_cat), self.log_diff, device=x_d.device)  #(batch_size, S)
                rows = torch.arange(batch_size, device=x_d.device) # (batch_size,)
                log_pi_ref[rows, x_d] = self.log_equal # (batch_size, S)
                
            if self.transition == 'gaussian':
                y_aux = torch.arange(self.n_cat, device=x_d.device)
                diff = x_d[:, None] - y_aux[None, :]
                exp_argument = -4*(diff)**2/((self.n_cat-1)**2 * self.beta)
                
                log_pi_vals = exp_argument - self.log_Z
                
                off_diag_mask = (diff != 0)

                off_diag_probs = torch.exp(log_pi_vals) * off_diag_mask.float()
                diag_probs = 1 - off_diag_probs.sum(dim=1, keepdim=True)
                
                diag_probs = torch.clamp(diag_probs, min=1e-8)
                diag_log = torch.log(diag_probs)
            
                log_pi_ref = torch.full((batch_size, self.n_cat), -1e8, device=self.device)  # -inf
                
                log_pi_ref[off_diag_mask] = log_pi_vals[off_diag_mask]
                
                rows = torch.arange(batch_size, device=self.device)
                log_pi_ref[rows, x_d] = diag_log.squeeze(1)
            
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner
            
        log_c = torch.logsumexp(self.log_alpha[None, :] + log_z, dim=1) #(K,) + (batch_size, K) -> (batch_size,)
        return log_c
        
    @torch.no_grad()
    def forward(self, x):

        num_samples = x.shape[0]
        log_z = torch.zeros(num_samples, self.n_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.dim):
            x_d        = x[:, d]
            if self.transition == 'uniform':
                log_pi_ref = torch.full((num_samples, self.n_cat), self.log_diff, device=x_d.device)  #(batch_size, S)
                rows = torch.arange(num_samples, device=x_d.device) # (batch_size,)
                log_pi_ref[rows, x_d] = self.log_equal # (batch_size, S)
                
            if self.transition == 'gaussian':
                y_aux = torch.arange(self.n_cat, device=x_d.device)
                diff = x_d[:, None] - y_aux[None, :]
                exp_argument = -4*(diff)**2/((self.n_cat-1)**2 * self.beta)
                
                log_pi_vals = exp_argument - self.log_Z
                off_diag_mask = (diff != 0)

                off_diag_probs = torch.exp(log_pi_vals) * off_diag_mask.float()
                diag_probs = 1 - off_diag_probs.sum(dim=1, keepdim=True)
                
                diag_probs = torch.clamp(diag_probs, min=1e-8)
                diag_log = torch.log(diag_probs)
            
                log_pi_ref = torch.full((num_samples, self.n_cat), -1e8, device=self.device)  # -inf
                log_pi_ref[off_diag_mask] = log_pi_vals[off_diag_mask]
                
                rows = torch.arange(num_samples, device=self.device)
                log_pi_ref[rows, x_d] = diag_log.squeeze(1)

            log_pi_ref_list.append(log_pi_ref)
                
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner # (batch_size, K)
        
        log_w_k = self.log_alpha[None, :] + log_z  # (K,) + (batch_size, K) -> (batch_size, K)
        
        log_p_k = log_w_k - torch.logsumexp(log_w_k, dim=1)[:, None] #(batch_size, K) - (batch_size, ) -> (batch_size, K)
        p_k = torch.exp(log_p_k) # (batch_size, K)
        k_stars = torch.multinomial(p_k, num_samples=1).squeeze(1)  # (batch_size,)
    
        y_samples = torch.zeros(num_samples, self.dim, dtype=torch.long, device=self.device)
    
        for d in range(self.dim):
            log_pi_ref = log_pi_ref_list[d]
                
            log_p_d_all = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(batch_size, K, S)
            batch_idx = torch.arange(num_samples, device=k_stars.device)
            log_p_d_selected = log_p_d_all[batch_idx, k_stars, :] #(batch_size, S)
            
            p_d = torch.softmax(log_p_d_selected, dim=1)
            y_d = torch.multinomial(p_d, num_samples=1).squeeze(1) #(batch_size,)
            y_samples[:, d] = y_d
        
        return y_samples
