import torch
import numpy as np
from matplotlib.figure import Figure
from PIL import Image
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt

def fig2img(fig: Figure) -> Image.Image:
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)
    buf = np.roll(buf, 3, axis = 2)
    return Image.frombytes("RGBA", (w, h), buf.tobytes())


def plot_samples(x, y, D, indices_plot, beta, n_steps, xrange=[0,100], yrange=[0,100], 
                 plot_trajectories=True, by_dims=False, index_0=0, index_1=1):
    xmin, xmax = xrange[0], xrange[1]
    ymin, ymax = yrange[0], yrange[1]
    num_trajectories = 5
    num_translations = 5
    dim    = x.shape[1]
    y_pred = D.sample(x).cpu()
    #x = x.cpu()
    #y = y.cpu()

    if dim > 2 and by_dims is False:
        pca = PCA(n_components=2)
        pca.fit(y.cpu())#(torch.cat([y, y_pred], dim=0))
        x_pca      = pca.transform(x.cpu())
        y_pca      = pca.transform(y.cpu())
        y_pred_pca = pca.transform(y_pred)
        index_0    = 0 
        index_1    = 1

    elif by_dims is False:
        
        pca        = None
        x_pca      = torch.clone(x).cpu()
        y_pca      = torch.clone(y).cpu()
        y_pred_pca = torch.clone(y_pred).cpu()
        index_0    = 0 
        index_1    = 1
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    traj_start = x[:num_trajectories]
    repeats = [num_translations] + [1] * traj_start.dim()
    traj_start = traj_start.unsqueeze(0).repeat(*repeats)
    traj_start = traj_start.reshape(-1, *x.shape[1:])
    


    axes[2].scatter(y_pred_pca[:, index_0], y_pred_pca[:, index_1], s=15, color='b', alpha=0.8, label=fr'$y \sim \pi(y|x)$')
    
    if plot_trajectories:
        trajectories = D.sample_trajectory(traj_start.cpu(), pca=pca)
        for i in indices_plot:
            axes[1].plot(
                [x_pca[i, 0], y_pca[i, 0]],
                [x_pca[i, 1], y_pca[i, 1]],
                color='g',
                linewidth=1.5,
                alpha=1.0
            )
        #axes[2].plot(
        #    [x_pca[i, 0].cpu(), y_pred_pca[i, 0]],
        #    [x_pca[i, 1].cpu(), y_pred_pca[i, 1]],
        #    color='g',
        #    linewidth=1.5,
        #    alpha=1.0
        #)
        axes[2].scatter(trajectories[0, :, 0], trajectories[0, :, 1], s=15, color='r', alpha=0.8, label=fr'$p_0(x)$')
        axes[2].scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], s=15, color='y', alpha=0.8)
        
        for i in range(num_trajectories * num_translations):
            axes[2].plot(trajectories[:, i, 0], trajectories[:, i, 1], color='g', linewidth=1.5, alpha=1.0)
            axes[2].plot(trajectories[:, i, 0], trajectories[:, i, 1], color='g', linewidth=1.5, alpha=1.0)
        
    axes[0].scatter(x_pca[:, index_0], x_pca[:, index_1], s=15, color='r', label=fr'$p_0(x)$')
    axes[1].scatter(y_pca[:, index_0], y_pca[:, index_1], s=15, color='b', label=fr'$p_1(y)$')

    axes[1].scatter(x_pca[indices_plot, index_0], x_pca[indices_plot, index_1], s=15, color='r', label=fr'$p_0(x)$')
    axes[1].scatter(y_pca[indices_plot, index_0], y_pca[indices_plot, index_1], s=15, color='y')
    
    #axes[2].scatter(x_pca[indices_plot, 0].cpu(), x_pca[indices_plot, 1].cpu(), s=15, color='r', label=fr'$p_0(x)$')
    #axes[2].scatter(y_pred_pca[indices_plot, 0].cpu(), y_pred_pca[indices_plot, 1].cpu(), s=15, color='y')
    
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.legend()
        ax.grid()

    fig.suptitle(fr"$\beta={beta}$, $n_s={n_steps}$", fontsize=16)


    if dim == 3:   
        fig2 = plt.figure(figsize=(15, 7))
        
        ax1 = fig2.add_subplot(121, projection='3d')
        ax1.scatter(
            y[:, 0].cpu(), 
            y[:, 1].cpu(), 
            y[:, 2].cpu(), 
            s=15, color='b', label=r'$p_0(x)$'
        )
        ax1.set_title('Original Data ($X0_{test}$)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Second subplot: Lines connecting X0_test to predicted points
        ax2 = fig2.add_subplot(122, projection='3d')
        indices_plot = torch.randint(0, len(x), (10,))  # Randomly pick 10 points
        
        # Plot connecting lines
        for i in indices_plot:
            ax2.plot(
                [x[i, 0].cpu(), y_pred[i, 0].cpu()],
                [x[i, 1].cpu(), y_pred[i, 1].cpu()],
                [x[i, 2].cpu(), y_pred[i, 2].cpu()],
                color='g',
                linewidth=1.5,
                alpha=1.0
            )
        
        # Scatter original (red) and predicted (yellow) points
        ax2.scatter(
            x[indices_plot, 0].cpu(), 
            x[indices_plot, 1].cpu(), 
            x[indices_plot, 2].cpu(), 
            s=15, color='r', label=r'$p_0(x)$'
        )
        ax2.scatter(
            y_pred[:, 0].cpu(), 
            y_pred[:, 1].cpu(), 
            y_pred[:, 2].cpu(), 
            s=15, color='y', label='Predicted'
        )
        ax2.set_title('Predictions vs. Ground Truth')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Set axis limits if needed
        zmin = ymin
        zmax = xmax
        for ax in [ax1, ax2]:
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_zlim(0, 100)  # Add z-axis limits if needed
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

    return fig

def plot_samples_benchmark(x, y, bench, indices_plot, beta, n_steps, xrange=[0,100], yrange=[0,100], 
                 plot_trajectories=True, by_dims=False, index_0=0, index_1=1):
    xmin, xmax = xrange[0], xrange[1]
    ymin, ymax = yrange[0], yrange[1]
    num_trajectories = 5
    num_translations = 4
    dim    = x.shape[1]
    y_pred = bench.sample_target_given_input(x, return_trajectories=False).cpu()
    #x = x.cpu()
    #y = y.cpu()

    if dim > 2 and by_dims is False:
        pca = PCA(n_components=2)
        pca.fit(y.cpu())#(torch.cat([y, y_pred], dim=0))
        x_pca      = pca.transform(x.cpu())
        y_pca      = pca.transform(y.cpu())
        y_pred_pca = pca.transform(y_pred)
        index_0    = 0 
        index_1    = 1

    elif by_dims is False:
        
        pca        = None
        x_pca      = torch.clone(x).cpu()
        y_pca      = torch.clone(y).cpu()
        y_pred_pca = torch.clone(y_pred).cpu()
        index_0    = 0 
        index_1    = 1
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    traj_start = x[:num_trajectories]
    repeats = [num_translations] + [1] * traj_start.dim()
    traj_start = traj_start.unsqueeze(0).repeat(*repeats)
    traj_start = traj_start.reshape(-1, *x.shape[1:])
    


    axes[2].scatter(y_pred_pca[:, index_0], y_pred_pca[:, index_1], s=15, color='b', alpha=0.8, label=fr'$y \sim \pi(y|x)$')
    
    if plot_trajectories:
        trajectories_stacked = bench.sample_target_given_input(traj_start.cpu(), return_trajectories=True)
        if dim > 2:
            trajectories = trajectories_stacked.reshape(-1, trajectories_stacked.shape[-1])
            trajectories = pca.transform(trajectories).reshape(2, -1, 2)
        else:
            trajectories = torch.clone(trajectories_stacked)
        for i in indices_plot:
            axes[1].plot(
                [x_pca[i, 0], y_pca[i, 0]],
                [x_pca[i, 1], y_pca[i, 1]],
                color='g',
                linewidth=1.5,
                alpha=1.0
            )
        #axes[2].plot(
        #    [x_pca[i, 0].cpu(), y_pred_pca[i, 0]],
        #    [x_pca[i, 1].cpu(), y_pred_pca[i, 1]],
        #    color='g',
        #    linewidth=1.5,
        #    alpha=1.0
        #)
        axes[2].scatter(trajectories[0, :, 0], trajectories[0, :, 1], s=15, color='r', alpha=0.8, label=fr'$p_0(x)$')
        axes[2].scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], s=15, color='y', alpha=0.8)
        
        for i in range(num_trajectories * num_translations):
            axes[2].plot(trajectories[:, i, 0], trajectories[:, i, 1], color='g', linewidth=1.5, alpha=1.0)
            axes[2].plot(trajectories[:, i, 0], trajectories[:, i, 1], color='g', linewidth=1.5, alpha=1.0)
        
    axes[0].scatter(x_pca[:, index_0], x_pca[:, index_1], s=15, color='r', label=fr'$p_0(x)$')
    axes[1].scatter(y_pca[:, index_0], y_pca[:, index_1], s=15, color='b', label=fr'$p_1(y)$')

    axes[1].scatter(x_pca[indices_plot, index_0], x_pca[indices_plot, index_1], s=15, color='r', label=fr'$p_0(x)$')
    axes[1].scatter(y_pca[indices_plot, index_0], y_pca[indices_plot, index_1], s=15, color='y')
    
    #axes[2].scatter(x_pca[indices_plot, 0].cpu(), x_pca[indices_plot, 1].cpu(), s=15, color='r', label=fr'$p_0(x)$')
    #axes[2].scatter(y_pred_pca[indices_plot, 0].cpu(), y_pred_pca[indices_plot, 1].cpu(), s=15, color='y')
    
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.legend()
        ax.grid()

    fig.suptitle(fr"$\beta={beta}$, $n_s={n_steps}$", fontsize=16)


    if dim == 3:   
        fig2 = plt.figure(figsize=(15, 7))
        
        ax1 = fig2.add_subplot(121, projection='3d')
        ax1.scatter(
            y[:, 0].cpu(), 
            y[:, 1].cpu(), 
            y[:, 2].cpu(), 
            s=15, color='b', label=r'$p_0(x)$'
        )
        ax1.set_title('Original Data ($X0_{test}$)')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        
        # Second subplot: Lines connecting X0_test to predicted points
        ax2 = fig2.add_subplot(122, projection='3d')
        indices_plot = torch.randint(0, len(x), (10,))  # Randomly pick 10 points
        
        # Plot connecting lines
        for i in indices_plot:
            ax2.plot(
                [x[i, 0].cpu(), y_pred[i, 0].cpu()],
                [x[i, 1].cpu(), y_pred[i, 1].cpu()],
                [x[i, 2].cpu(), y_pred[i, 2].cpu()],
                color='g',
                linewidth=1.5,
                alpha=1.0
            )
        
        # Scatter original (red) and predicted (yellow) points
        ax2.scatter(
            x[indices_plot, 0].cpu(), 
            x[indices_plot, 1].cpu(), 
            x[indices_plot, 2].cpu(), 
            s=15, color='r', label=r'$p_0(x)$'
        )
        ax2.scatter(
            y_pred[:, 0].cpu(), 
            y_pred[:, 1].cpu(), 
            y_pred[:, 2].cpu(), 
            s=15, color='y', label='Predicted'
        )
        ax2.set_title('Predictions vs. Ground Truth')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_zlabel('Z')
        
        # Set axis limits if needed
        zmin = ymin
        zmax = xmax
        for ax in [ax1, ax2]:
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_zlim(0, 100)  # Add z-axis limits if needed
            ax.legend()
            ax.grid(True)
        
        plt.tight_layout()
        plt.show()

    return fig

class Sampler:
    def __init__(
        self, device='cuda',
    ):
        self.device = device
    
    def sample(self, size=5):
        pass
        
class LoaderSampler(Sampler):
    def __init__(self, loader, device='cpu'):
        super(LoaderSampler, self).__init__(device)
        self.loader = loader
        self.it = iter(self.loader)
        
    def __len__(self):
        return len(self.loader)
    
    def reset_sampler(self):
        self.it = iter(self.loader)
        
    def sample(self, size=5):
        if size <= self.loader.batch_size:
            try:
                batch = next(self.it)
            except StopIteration:
                self.it = iter(self.loader)
                return self.sample(size)
            if len(batch) < size:
                return self.sample(size)
                
            return batch[:size].to(self.device)
            
        elif size > self.loader.batch_size:
            samples = []
            cur_size = 0
            
            while cur_size < size:
                try:
                    batch = next(self.it)
                    samples.append(batch)
                    cur_size += batch.shape[0]
                except StopIteration:
                    self.it = iter(self.loader)
                    print(f'Maximum size allowed exceeded, returning {cur_size} samples...')
                    samples = torch.cat(samples, dim=0)
                    return samples[:cur_size].to(self.device)
                
            samples = torch.cat(samples, dim=0)
            return samples[:size].to(self.device)

def sample_separated_means(num_potentials, dim, num_categories, min_dist=5, max_attempts=5000):
    means = []
    attempts = 0
    low, high = 5, num_categories - 5

    while len(means) < num_potentials and attempts < max_attempts:
        candidate = torch.randint(low, high, (dim,))
        if all(torch.norm(candidate - m.float()) >= min_dist for m in means):
            means.append(candidate)
        attempts += 1

    if len(means) < num_potentials:
        raise RuntimeError(f"Could only generate {len(means)} points with min_dist={min_dist}")
    
    return torch.stack(means)