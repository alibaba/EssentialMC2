# [Self-supervised Motion Learning from Static Images](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Self-Supervised_Motion_Learning_From_Static_Images_CVPR_2021_paper.pdf)

## Introduction

This work proposes MoSI, a simple framework for the video models to learn motion representations from images. It is
shown that MoSI can discover and attend to prominent motions in videos, thus yielding a strong representation for the
downstream action recognition task.

![流程](papers/CVPR2021-MOSI/resources/procedure.jpg)

## Example usage

#### Usage

* Run

```
# We recommend run pretrain with 16 V100 cards, and fintune with 8 V100 cards.
# Pretrain ResNet3D-R2D3DBranch on Hmdb51 dataset
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 run_train.py \
    --config config/MoSI_r2d3d_hmdb.py --dist_launcher pytorch
# Finetune ResNet3D-R2D3DBranch on Hmdb51 dataset
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 run_train.py \
    --config config/Finetunue_r2d3d_hmdb.py --dist_launcher pytorch
```
