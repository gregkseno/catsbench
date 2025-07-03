import os
from typing import List, Literal, Optional
import json
import wandb

from lightning.pytorch import Callback, Trainer, LightningModule
from lightning.pytorch.utilities import rank_zero_only

from src.metrics import FID, CMMD, GenerativeNLL, ClassifierAccuracy
from src.metrics import MSE, LPIPS, HammingDistance, EditDistance, BLEUScore


def visualize_texts(
    x_end: List[str], 
    x_start: List[str], 
    pred_x_start: List[str], 
    fb: Literal['forward', 'backward'],
    labels: Optional[List[str]] = [r'$p_0$', r'$p_1$', r'$p_{\theta}$'],
    iteration: Optional[int] = None, 
    exp_path: Optional[str] = None, 
    tracker: Optional[GeneralTracker] = None,
    step: Optional[int] = None
):
    sample_triplets = list(zip(x_end, pred_x_start, x_start))
    if exp_path is not None:
        jsonl_path = os.path.join(exp_path, 'samples', f'samples_{fb}_{iteration}_step_{step}.jsonl')
        if not os.path.isfile(jsonl_path):
            with open(jsonl_path, 'w') as f:
                for init, generated, example in sample_triplets:
                    sample = {
                        "initial": init,
                        "generated": generated,
                        "example": example,
                    }
                    f.write(json.dumps(sample) + '\n')
                    
    if tracker:
        table = wandb.Table(columns=["Initial text", "Generated text", "Example text"])
        for init, generated, example in sample_triplets:
            table.add_data(init, generated, example)
        tracker.log({f'{fb}_text_samples': table}, step=step)