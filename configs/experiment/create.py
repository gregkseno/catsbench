import os
import yaml

def make_config(dim, prior_type, alpha, num_timesteps, num_skip_steps,
                kl_loss_coeff, mse_loss_coeff):
    return {
        "defaults": [
            "csbm/benchmark/default"
        ],
        "data": {
            "dim": dim,
        },
        "prior": {
            "prior_type": prior_type,
            "alpha": alpha,
            "num_timesteps": num_timesteps,
            "num_skip_steps": num_skip_steps,
        },
        "method": {
            "kl_loss_coeff": kl_loss_coeff,
            "mse_loss_coeff": mse_loss_coeff,
        },
    }

# --- parameter grid ---
dims = [2, 16, 64]

priors = {
    "gaussian": [0.02, 0.05],
    "uniform": [0.005, 0.01],
}

timesteps = [
    (63, 2),
    (15, 8),
]

loss_coeffs = [
    (1.0, 0.0),  # KL
    (0.0, 1.0),  # MSE
]

# --- output folder ---
out_dir = "configs/experiment/csbm/benchmark"
os.makedirs(out_dir, exist_ok=True)

# --- generate ---
for dim in dims:
    for prior_type, alphas in priors.items():
        for alpha in alphas:
            for num_timesteps, num_skip_steps in timesteps:
                for kl, mse in loss_coeffs:
                    cfg = make_config(dim, prior_type, alpha,
                                      num_timesteps, num_skip_steps,
                                      kl, mse)

                    loss_tag = "kl" if kl == 1.0 else "mse"
                    alpha_tag = str(alpha).replace(".", "")
                    fname = f"d{dim}_{prior_type[0]}{alpha_tag}_t{num_timesteps}_{loss_tag}.yaml"

                    path = os.path.join(out_dir, fname)
                    with open(path, "w") as f:
                        f.write("# @package _global_\n\n")
                        yaml.dump(cfg, f, sort_keys=False)

print(f"✅ Generated configs in {out_dir}")
