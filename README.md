# EssentialMC2

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/essmc2)](https://pypi.org/project/essmc2/) [![PyPI](https://img.shields.io/pypi/v/essmc2)](https://pypi.org/project/essmc2) [![license](https://img.shields.io/github/license/alibaba/EssentialMC2.svg)](https://github.com/alibaba/EssentialMC2/blob/main/LICENSE)

### Introduction

EssentialMC2 is a complete system to solve video understanding tasks including MHRL(representation learning), MECR2(
relation reasoning) and MOSL3(openset life-long learning) powered by [DAMO Academy](https://damo.alibaba.com/?lang=en)
MinD(Machine IntelligenNce of Damo) Lab. This codebase provides a comprehensive solution for video classification, 
temporal detection and noise learning.

### Features

- Simple and easy to use
- High efficiency
- Include SOTA papers presented by DAMO Academy
- Include various pretrained models

### Installation

#### Install by pip

Run `pip install essmc2`.

#### Install from source

##### Requirements

* Python 3.6+
* PytTorch 1.5+

Run `python setup.py install`. For each specific task, please refer to task specific README.

### Model Zoo

Pretrained models can be found in the [MODEL_ZOO.md](MODEL_ZOO.md).

### SOTA Tasks

- TAda! Temporally-Adaptive Convolutions for Video Understanding <br>
[[Project](https://github.com/alibaba-mmai-research/TAdaConv/blob/main/projects/tada/README.md)] [[Paper](https://arxiv.org/pdf/2110.06178.pdf)] [[Website](https://tadaconv-iclr2022.github.io)] **ICLR 2022**
- NGC: A Unified Framework for Learning with Open-World Noisy Data <br>
[[Project](papers/ICCV2021-NGC/README.md)] [[Paper](https://arxiv.org/abs/2108.11035)] **ICCV 2021**
- Self-supervised Motion Learning from Static Images <br>
[[Project](papers/CVPR2021-MOSI/README.md)] [[Paper](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Self-Supervised_Motion_Learning_From_Static_Images_CVPR_2021_paper)] **CVPR 2021**
- A Stronger Baseline for Ego-Centric Action Detection <br>
[[Project](https://github.com/alibaba-mmai-research/TAdaConv/blob/main/projects/epic-kitchen-tal/README.md)] [[Paper](https://arxiv.org/pdf/2106.06942)] 
**First-place** submission to [EPIC-KITCHENS-100 Action Detection Challenge](https://competitions.codalab.org/competitions/25926#results)
- Towards Training Stronger Video Vision Transformers for EPIC-KITCHENS-100 Action Recognition <br>
[[Project](https://github.com/alibaba-mmai-research/TAdaConv/blob/main/projects/epic-kitchen-ar/README.md)] [[Paper](https://arxiv.org/pdf/2106.05058)] 
**Second-place** submission to [EPIC-KITCHENS-100 Action Recognition challenge](https://competitions.codalab.org/competitions/25923#results)

### License

EssentialMC2 is released under [MIT license](https://github.com/alibaba/EssentialMC2/blob/main/LICENSE).

```text
MIT License

Copyright (c) 2021 Alibaba

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Acknowledgement

EssentialMC2 is an open source project that is contributed by researchers from DAMO Academy. We appreciate users who
give valuable feedbacks.
