# CausalFM


🚨 **New Release** 🚨  

We have released a new package **CausalFM-toolkit** with a clean API and ready-to-use implementations:  

👉 GitHub: https://github.com/yccm/CausalFM-toolkit

📖 Docs: https://causalfm-toolkit.readthedocs.io

(Recommended for users who want to directly apply CausalFM in practice)




## 📌 Introduction

PyTorch Implementation on Paper [Foundation Models for Causal Inference via Prior-Data Fitted Networks](https://arxiv.org/abs/2506.10914)


In this paper, we introduce **CausalFM**, a comprehensive framework for training PFN-based foundation models in various causal inference settings.


CausalFM provides a **unified framework** for training foundation models across multiple causal inference tasks, including:  

- **Standard CATE estimation setting**  
- **Instrumental Variables (IV) setting**  
- **Front-door adjustment setting**  

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


## 📊 Data Generation

We provide scripts to generate training datasets for various causal inference settings:


Standard CATE

```
cd DATA_standard
python gen_standard_syn.py 
```
Instrumental Variables (IV) 

```
cd DATA_IV
python gen_iv_data_binary.py # Binary Instrument
python gen_iv_data_conti.py # Continuous Instrument
```


Front-door adjustment

```
cd DATA_FD
python gen_frontdoor.py
```

## 🎯 Training

Standard CATE
```
python src/tabpfn/train_standard/training_standard.py 
```

Instrumental Variables (IV)
```
python src/tabpfn/train_iv/training_iv_binary.py
python src/tabpfn/train_iv/training_iv_conti.py
 
```
Front-door adjustment
```
python src/tabpfn/train_fd/training_fd.py
```

## 📈 Evaluation
```
├── evaluation/notebook/
│   ├── test_fd.ipynb            # Jupyter notebook: FD evaluation
│   ├── test_iv_binary.ipynb     # Jupyter notebook: Binary IV evaluation
│   ├── test_iv_conti.ipynb      # Jupyter notebook: Continuous IV evaluation
│   ├── test_jobs.ipynb          # Jupyter notebook: Jobs dataset evaluation
│   └── test_standard_cate.ipynb # Jupyter notebook: Standard CATE evaluation
```

## 📖 Citation

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
