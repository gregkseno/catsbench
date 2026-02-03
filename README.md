<div align="center">

# Entering the Era of Discrete Diffusion Models: A Benchmark for Schrödinger Bridges and Entropic Optimal Transport

[Xavier Aramayo](https://scholar.google.com/citations?user=1B9UIYoAAAAJ),
[Grigoriy Ksenofontov](https://scholar.google.com/citations?user=e0mirzYAAAAJ), [Aleksei Leonov](https://scholar.google.com/citations?user=gzj9nOcAAAAJ), [Iaroslav Koshelev](https://scholar.google.com/citations?user=gmaJRL4AAAAJ), [Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ)

[![arXiv Paper](https://img.shields.io/badge/arXiv-2502.01416-b31b1b)](https://arxiv.org/abs/2502.01416)
[![OpenReview Paper](https://img.shields.io/badge/OpenReview-PDF-8c1b13)](https://openreview.net/forum?id=XcPDT615Gd)
[![GitHub](https://img.shields.io/github/stars/gregkseno/catsbench?style=social)](https://github.com/gregkseno/catsbench)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-view-green)](https://huggingface.co/gregkseno/catsbench)
![GitHub License](https://img.shields.io/github/license/gregkseno/csbm)

</div>

This repository contains the official implementation of the paper *"Entering the Era of Discrete Diffusion Models: A Benchmark for Schrödinger Bridges and Entropic Optimal Transport"*, accepted at **ICLR 2026**.

## 📌 TL;DR

This paper proposes a benchmark for Schrödinger Bridge (SB) and entropic optimal transport (EOT) methods on discrete spaces, and adapts several continuous SB/EOT approaches to the discrete setting.

<!-- ![teaser](./images/teaser.png) -->

## 📦 CatSBench (Package)

`catsbench` is the standalone benchmark package. It provides benchmark definitions, evaluation metrics, and reusable utilities, including a Triton-optimized log-sum-exp (LSE) matmul kernel.

### 📥 Installation

Install the benchmark package via `pip`:

```bash
pip install catsbench
```

### 🚀 Quickstart

Load a benchmark definition and its assets from a pretrained repository:

```python
from catsbench import BenchmarkHD

bench = BenchmarkHD.from_pretrained(
    "gregkseno/catsbench",
    "hd_d2_s50_gaussian_a0.02_gaussian",
    init_benchmark=False,  # skip heavy initialization at load time
)
```

To sample marginals $p_0$ and $p_1$:

```python
x_start, x_end = bench.sample_input_target(32) # ([B=32, D=2], [B=32, D=2])
```

Or sample them separately:

```python
x_start = bench.sample_input(32) # [B=32, D=2]
x_end = bench.sample_target(32)  # [B=32, D=2]
```

> [!IMPORTANT]
> Both examples above sample independently, i.e., $(x_0, x_1) \sim p_0(x_0),p_1(x_1)$.

To sample from the ground-truth EOT/SB coupling, i.e., $(x_0, x_1) \sim p_0(x_0),q^*(x_1 | x_0)$, use:

```python
x_start = bench.sample_input(32) # [B=32, D=2]
x_end = bench.sample(x_start)    # [B=32, D=2]
```

> [!NOTE]
> See the end-to-end benchmark workflow (initialization, evaluation, metrics, plotting) in `notebooks/benchmark_usage.ipynb`

-----

## Reproducing Experiments

This part describes how to run the full training and evaluation pipeline to reproduce paper's results. It explains how to launch experiments for the provided methods (DLightSB, DLightSB-M, CSBM, $\alpha$-CSBM) and evaluate them on the benchmarks.

```bash
|-- configs
|   |-- config.yaml   # main Hydra entrypoint
|   |-- callbacks     # Lightning callbacks: benchmark metrics + visualization
|   |-- data          # datamodule/dataset configs
|   |-- experiment    # experiment presets (override bundles)
|   |-- hydra         # Hydra runtime/output settings
|   |-- logger        # logging backends (Comet, W&B, TensorBoard)
|   |-- method        # method-level configs (e.g., CSBM, DLightSB)
|   |-- model         # model architecture configs
|   |-- prior         # reference process configs
|   `-- trainer       # trainer, hardware, precision, runtime configs
|-- logs              # logs, checkpoints, and run artifacts
|-- notebooks         # analysis and baselines
|-- scripts           # bash (+ SLURM) launch scripts
`-- src
    |-- catsbench     # benchmark package code
    |-- data          # Lightning datamodules + reference process implementation
    |-- methods       # training/inference methods (e.g., CSBM, DLightSB)
    |-- metrics       # callbacks computing benchmark metrics
    |-- plotter       # callbacks for plotting samples and trajectories
    |-- utils         # instantiation, logging, common helpers
    `-- run.py        # main entrypoint for training and testing
```

### 📦 Dependencies

Create the Anaconda environment using the following command:

```bash
conda env update -f environment.yml
```

and activate it:

```bash
conda activate catsbench
```

### 🏋️ Training

To start training, pick an experiment config under `configs/experiment/<method_name>/benchmark_hd/<exp_name>.yaml` and launch it with:

```bash
python -m src.run experiment=<method_name>/benchmark_hd/<exp_name>
```

> [!TIP]Example
>
> ```bash
> python -m src.run experiment=dlight_sb/benchmark_hd/d2_g002
> ```

### 📊 Evaluation

Use the same experiment config as in training and set a checkpoint:

- Manual path: `logs/runs/<method_name>/benchmark_hd/<exp_name>/<seed>/<date>/epoch_<...>.ckpt`
- Or set `ckpt_path=auto` to automatically load the latest checkpoint based on the config.

```bash
python -m src.run task_name=test ckpt_path=auto \
  experiment=<method_name>/benchmark_hd/<exp_filename>
```

> [!TIP]Example
>
> ```bash
> python -m src.run task_name=test ckpt_path=auto \
>   experiment=dlight_sb/benchmark_hd/d2_g002
> ```


### 🎓 Citation

```bibtex
@misc{
  carrasco2025enteringeradiscretediffusion,
  title={Entering the Era of Discrete Diffusion Models: A Benchmark for {Schr\"odinger} Bridges and Entropic Optimal Transport}, 
  author={Xavier Aramayo Carrasco and Grigoriy Ksenofontov and Aleksei Leonov and Iaroslav Sergeevich Koshelev and Alexander Korotin},
  year={2025},
  eprint={2509.23348},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2509.23348}, 
}
```

## 🙏 Credits

- [Comet ML](https://www.comet.com) — experiment-tracking and visualization toolkit;
- [Inkscape](https://inkscape.org/) — an excellent open-source editor for vector graphics;
- [Hydra/Lightning template](https://github.com/ashleve/lightning-hydra-template) - project template used as a starting point.
