# CausalFM

PyTorch Implementation on Paper [Foundation Models for Causal Inference via Prior-Data Fitted Networks](https://arxiv.org/abs/2506.10914)

## Introduction

In this paper, we introduce **CausalFM**, a comprehensive framework for training PFN-based foundation models in various causal inference settings.


CausalFM provides a **unified framework** for training foundation models across multiple causal inference tasks, including:  

- **Standard CATE estimation**  
- **Instrumental Variables (IV) learning**  
- **Front-door adjustment**  

This repository contains dataset generation pipelines, model implementations, and training/evaluation scripts.



---

## 🚀 Installation

Clone the repository and install dependencies:

```
git clone https://github.com/yccm/CausalFM.git
cd CausalFM
conda create -n causalfm
conda activate causalfm
pip install -r requirements.txt
```

## Data Generation

We provide scripts to generate training datasets for various causal inference settings:

### Standard CATE

```
cd DATA_standard
python gen_standard_syn.py 
```
### IV setting

### FD setting


## Training


## Evaluation

## Citation

If you find this repository useful, please cite our paper:

```
@article{ma2025foundation,
  title={Foundation Models for Causal Inference via Prior-Data Fitted Networks},
  author={Ma, Yuchen and Frauen, Dennis and Javurek, Emil and Feuerriegel, Stefan},
  journal={arXiv preprint arXiv:2506.10914},
  year={2025}
}
```


## Acknowledgement
This repo is based on the implementation of [TabPFN](https://github.com/PriorLabs/TabPFN/) 
