from math import log
import torch.nn as nn
import torch
from ..data import prior

def create_dimensional_points(d, min_val=10.0, max_val=40.0, device='cpu'):

    base_points = torch.tensor([
        [min_val, min_val, max_val, max_val],  
        [min_val, max_val, max_val, min_val]   
    ], device=device)
    
    full_points = torch.full((d, 4), min_val, device=device)
    full_points[:2] = base_points
    
    if d >= 3:
        
        full_points[2, 1] = max_val  
        full_points[2, 2] = max_val  
    
    return full_points
    
class LightSB_D(nn.Module):
    def __init__(self, dim, n_cat, n_potentials, n_steps=1, beta=1e-5, device='cuda', distr_init='randn', prior_type='uniform'):
        super().__init__()
        self.dim = dim
        self.n_cat = n_cat
        self.n_potentials = n_potentials
        self.device = device
        self.n_steps = n_steps
        
        self.log_alpha = nn.Parameter(torch.zeros(n_potentials, device=device))
        #self.log_cp_cores = nn.ParameterList([nn.Parameter(torch.zeros(n_potentials, n_cat, device=device)) for _ in range(dim)])
        self.prior_type = prior_type
        self.beta  = beta
        self.distr_init = distr_init
        
        self._initialize_parameters(distr_init)

        self.prior = prior.Prior(beta, n_cat, n_steps, num_skip_steps=1, prior_type=prior_type).to(device)

            
    def _initialize_parameters_old(self, distr_init):
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)
        for core in self.log_cp_cores:
            if distr_init == 'gaussian':
                nn.init.normal_(core, mean=-1.0, std=0.5)
            elif distr_init == 'uniform':
                nn.init.constant_(core, -torch.log(torch.tensor(self.n_cat * 1.0)))
            else:
                raise ValueError(f"Invalid distr_init: {distr_init}")

    def _initialize_parameters(self, distr_init):
        nn.init.normal_(self.log_alpha, mean=-2.0, std=0.1)

        self.log_cp_cores = []
        
        if self.distr_init == 'poisson':
            rates = create_dimensional_points(self.dim, 10, self.n_cat-10).to(self.device)
            y_d   = torch.arange(self.n_cat, device=self.device)  #(D, S)
            
        for d in range(self.dim):
        
            if self.distr_init == 'gaussian':
                cur_core = (-1.0 + 0.5**2*torch.randn(self.n_potentials, self.n_cat))/(self.n_cat * self.n_potentials)
                cur_log_core = torch.log(cur_core**2)
                
            elif self.distr_init == 'uniform':
                cur_log_core = torch.log(torch.ones(self.n_potentials, self.n_cat)/(self.n_cat * self.n_potentials))
                    
            elif self.distr_init == 'poisson':
                rate = rates[d] # (K,)
                cur_log_core = y_d[None, :] * torch.log(rate[:, None]) - rate[:, None] - torch.lgamma(y_d[None, :] + 1) #(K, S)
                
            cur_log_core = cur_log_core.to(self.device)
            self.log_cp_cores.append(nn.Parameter(cur_log_core))

        self._make_model_parameters()

    def _make_model_parameters(self):
        parameters = []

        for core in self.log_cp_cores:
            parameters.append(core)

        self.parameters = nn.ParameterList(parameters)
                
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
            
            t = torch.randint(low=1, high=self.n_steps + 2, size=(batch_size,), device=self.device)
            log_pi_ref = self.prior.extract('cumulative', t, row_id=x_d)
            
            log_joint = self.log_cp_cores[d][None, :, :] + log_pi_ref[:, None, :] #(K, S) + (batch_size, S) -> (batch_size, K, S)
            log_inner = torch.logsumexp(log_joint, dim=2)  # (batch_size, K)
            log_z = log_z + log_inner
            
        log_c = torch.logsumexp(self.log_alpha[None, :] + log_z, dim=1) #(K,) + (batch_size, K) -> (batch_size,)
        return log_c
        
    @torch.no_grad()
    def sample(self, x):

        num_samples = x.shape[0]
        log_z = torch.zeros(num_samples, self.n_potentials, device=self.device)
        log_pi_ref_list = []
        for d in range(self.dim):
            x_d        = x[:, d]
            t = torch.randint(low=1, high=self.n_steps + 2, size=(num_samples,), device=self.device)
            log_pi_ref = self.prior.extract('cumulative', t, row_id=x_d)

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

    def get_log_probs(self):
        if self.dist_type == 'categorical':
            probs = F.softmax(self.cp_cores, dim=-1)
        
        elif self.dist_type == 'poisson_old':
            rates = torch.tensor() 
            y = torch.arange(self.n_cat, device=rates.device)
            log_probs = y * torch.log(rates.unsqueeze(-1)) - rates.unsqueeze(-1)
            log_probs -= torch.lgamma(y + 1)
            probs = torch.exp(log_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)

        elif self.dist_type == 'poisson':
            
            rates = create_dimensional_points(self.dim, 10, self.n_cat-10).to(self.device)
            
            y     = [torch.arange(self.n_cat, device=self.device) for _ in range(self.dim)]
            
            for d in range(self.dim):
                y_d  = y[d]     
                rate = rates[d] 
                log_prob_d = y_d[None, :] * torch.log(rate[:, None]) - rate[:, None] - torch.lgamma(y_d[None, :] + 1) 
                self.log_cp_cores.append(log_prob_d)

        elif self.dist_type == 'negbinomial':
            r = 1 + 9 * torch.sigmoid(self.r_r)  
            p = torch.sigmoid(self.r_p)          
            y = torch.arange(self.n_cat, device=r.device)

            log_binom = torch.lgamma(y + r.unsqueeze(-1)) - torch.lgamma(r.unsqueeze(-1))
            log_binom -= torch.lgamma(y + 1)
            log_probs = log_binom + r.unsqueeze(-1) * torch.log(p.unsqueeze(-1))
            log_probs += y * torch.log(1 - p.unsqueeze(-1))
            probs = torch.exp(log_probs)
            probs = probs / probs.sum(dim=-1, keepdim=True)
        
        elif self.dist_type == 'bernoulli':
            p = torch.sigmoid(self.cp_cores)  
            probs = torch.stack([1 - p, p], dim=-1)
