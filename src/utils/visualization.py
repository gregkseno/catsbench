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

def plot_samples(x, y, D, indices_plot, beta, n_steps, xrange=[0,100], yrange=[0,100], plot_trajectories=True):
    xmin, xmax = xrange[0], xrange[1]
    ymin, ymax = yrange[0], yrange[1]
    num_trajectories = 5
    num_translations = 5
    dim = x.shape[1]
    y_pred = D.sample(x).cpu()
    x = x.cpu()
    y = y.cpu()
    
    if dim > 2:
        pca = PCA(n_components=2)
        pca.fit(y)
        x_pca      = pca.transform(x)
        y_pca      = pca.transform(y)
        y_pred_pca = pca.transform(y_pred)

    else:
        pca = None
        x_pca      = torch.clone(x).cpu()
        y_pca      = torch.clone(y).cpu()
        y_pred_pca = torch.clone(y_pred).cpu()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    traj_start = x[:num_trajectories]
    repeats = [num_translations] + [1] * traj_start.dim()
    traj_start = traj_start.unsqueeze(0).repeat(*repeats)
    traj_start = traj_start.reshape(-1, *x.shape[1:])
    trajectories = D.sample_trajectory(traj_start, pca=pca)

    
        
    #axes[2].scatter(y_pred_pca[:, 0], y_pred_pca[:, 1], s=15, color='b', alpha=0.8, label=fr'$y \sim \pi(y|x)$')

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
        
    axes[2].scatter(y_pred_pca[:, 0], y_pred_pca[:, 1], s=15, color='b', alpha=0.8, label=fr'$y \sim \pi(y|x)$')
    
    if plot_trajectories:
        axes[2].scatter(trajectories[0, :, 0], trajectories[0, :, 1], s=15, color='r', alpha=0.8, label=fr'$p_0(x)$')
        axes[2].scatter(trajectories[-1, :, 0], trajectories[-1, :, 1], s=15, color='y', alpha=0.8)
        
        for i in range(num_trajectories * num_translations):
            axes[2].plot(trajectories[:, i, 0], trajectories[:, i, 1], color='g', linewidth=1.5, alpha=1.0)
            axes[2].plot(trajectories[:, i, 0], trajectories[:, i, 1], color='g', linewidth=1.5, alpha=1.0)
        
    axes[0].scatter(x_pca[:, 0], x_pca[:, 1], s=15, color='r', label=fr'$p_0(x)$')
    axes[1].scatter(y_pca[:, 0], y_pca[:, 1], s=15, color='b', label=fr'$p_1(y)$')

    axes[1].scatter(x_pca[indices_plot, 0], x_pca[indices_plot, 1], s=15, color='r', label=fr'$p_0(x)$')
    axes[1].scatter(y_pca[indices_plot, 0], y_pca[indices_plot, 1], s=15, color='y')
    
    #axes[2].scatter(x_pca[indices_plot, 0].cpu(), x_pca[indices_plot, 1].cpu(), s=15, color='r', label=fr'$p_0(x)$')
    #axes[2].scatter(y_pred_pca[indices_plot, 0].cpu(), y_pred_pca[indices_plot, 1].cpu(), s=15, color='y')
    
    for ax in axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.legend()
        ax.grid()

    fig.suptitle(fr"$\beta={beta}$, $n_s={n_steps}$", fontsize=16)
    return fig