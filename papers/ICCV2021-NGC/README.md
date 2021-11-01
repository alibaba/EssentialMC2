# [NGC: A Unified Framework for Learning with Open-World Noisy Data](https://arxiv.org/abs/2108.11035)

## Introduction

We propose a new graph-based framework, namely Noisy Graph Clean- ing (NGC), which collects clean samples by leveraging
ge- ometric structure of data and model predictive confidence.

![流程](resources/procedure.jpg)

## Results

Hyper parameters can be found in configs folder.

## Example usage

#### Requirements and Installation

* Install requirements by

```
pip install -r requirements.txt
```

* Build c++ library

```
cd impls
python setup.py build_ext -i
cd ..
```

#### Usage

* Run

```
python run.py --config config/cifar10_sym.py --work_dir work_cifar10_sym
```

* Resume from checkpoint

```
python run.py --config config/cifar10_sym.py --work_dir work_cifar10_sym --resume_from work_cifar10_sym/epoch-20.pth
```

* Change random seed

```
python run.py --config config/cifar10_sym.py --work_dir work_cifar10_sym --seed 1001
```
