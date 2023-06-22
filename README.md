# TorchLens - Ray Tracing Library for PyTorch


## This repository is a PyTorch implementation of the joint-lens-design repository by Côté, Mannan, et.al.
### [Paper](https://arxiv.org/abs/2212.04441) | [Project Page](https://light.princeton.edu/joint-lens-design)
Authors of the original repository.
https://github.com/gffrct/joint-lens-design.git
#### Geoffroi Côté, Fahim Mannan, Simon Thibault, Jean-François Lalonde, Felix Heide

## Requirements

The file ```environment.yml``` can be used to install a functional Conda environment. The environment was tested on Python 3.10 and Tensorflow 2.8, but any recent version of these packages should suffice.

```
conda env create -n joint-lens-design -f environment.yml
conda activate joint-lens-design
```

## Simulating aberrations

The sample script ```simulate_aberrations.py``` provides a simple demonstration of the proposed method. We provide four ```.yml``` files to model spherical lenses with 1, 2, 3, and 4 refractive elements. The latter three correspond to the baseline lenses in the paper. For a complete list of command-line arguments, try:

```
python simulate_aberrations --help
```

For any question or advice, please reach out to me at haruka_takahira[at]keio[dot]jp.
