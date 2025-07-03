<div align="center">

# Discrete Entropic Optimal Transport Benchmark for Generative Modeling

[Xavier Aramayo](https://scholar.google.com/citations?user=1B9UIYoAAAAJ),
[Grigoriy Ksenofontov](https://scholar.google.com/citations?user=e0mirzYAAAAJ),
[Alexander Korotin](https://scholar.google.ru/citations?user=1rIIvjAAAAAJ)

<!-- [![arXiv Paper](https://img.shields.io/badge/arXiv-2502.01416-b31b1b)](https://arxiv.org/abs/2502.01416)
[![OpenReview Paper](https://img.shields.io/badge/OpenReview-PDF-8c1b13)](https://openreview.net/forum?id=RBly0nOr2h)
[![GitHub](https://img.shields.io/github/stars/gregkseno/csbm?style=social)](https://github.com/gregkseno/csbm)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-view-green)](https://huggingface.co/gregkseno/csbm)
[![WandB](https://img.shields.io/badge/W%26B-view-green)](https://wandb.ai/gregkseno/csbm) -->
![GitHub License](https://img.shields.io/github/license/gregkseno/csbm)

</div>

This repository contains the official implementation of the paper "???", accepted at ???.

## 📌 TL;DR

This paper proposes the Schrödinger Bridge problem to work with discrete time and spaces.

<!-- ![teaser](./images/teaser.png) -->

## 🗂️ Repository structure

```bash
|-- configs     # hydra configs
|-- logs        # experiment logs
|-- notebooks   # experiments & analysis
`-- src         
    |-- bench   # bencmark package code
    |-- data    # lightning datamodules
    |-- methods # e.g. CSBM, DDSBM, etc.
    |-- metrics # non-benchmark metrics
    |-- utils   # e.g. for logging, data
    `-- run.py  # main entrypoint for training & testing
```

## 📦 Dependencies

Create the Anaconda environment using the following command:

```bash
conda env update -f environment.yml
```

## 🛠️ Preparations

### Download Datasets

## 🏋️‍♂️ Training

## 📊 Evaluation

## 🎓 Citation

```bibtex
@article{
  ...
}
```

## 🙏 Credits

- [Weights & Biases](https://wandb.ai) — experiment-tracking and visualization toolkit;
- [Inkscape](https://inkscape.org/) — an excellent open-source editor for vector graphics;
- [Hydra/Lightning template](https://github.com/ashleve/lightning-hydra-template) - cool.
