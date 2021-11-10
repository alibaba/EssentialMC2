# MODEL ZOO

## Kinetics

| Dataset | architecture | depth | init | clips x crops | #frames x sampling rate | acc@1 | acc@5 | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| K400 | TAda2D | R50 | IN-1K | 10 x 3 | 8 x 8 | 76.3 | 92.4 | [[google drive](https://drive.google.com/file/d/1-_Yek3lFYpgahU2Q2ITON4rgqm5Voj2E/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1CzJ82frDeH8P4L55aZdZ-Q)(code:jp7d)] | configs/models/tada_r50_k400.py |
| K400 | TAda2D | R50 | IN-1K | 10 x 3 | 16 x 5 | 76.9 | 92.7 | [[google drive](https://drive.google.com/file/d/1Phj2mSna27Gv46tJzjw4a51_qum1HW6x/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1GQZmB_ZKwq1NjmVooICa2w)(code:buww)] | configs/models/tada_r50_k400.py |
| K400 | ViViT Fact. Enc. | B16x2 | IN-21K | 4 x 3 | 32 x 2 | 79.4 | 94.0 | [[google drive](https://drive.google.com/file/d/1xD4uij9DmZojnl1xuWBa-gwm5hUZxDc7/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1iVjKjEMm-6ymUd15ZNqvXw)(code:1t51)] | configs/models/vivit_fac_enc_b16x2_k400.py |

## Something-Something
| Dataset | architecture | depth | init | clips x crops | #frames | acc@1 | acc@5 | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| SSV2 | TAda2D | R50 | IN-1K | 2 x 3 | 8 | 63.8 | 87.7 | [[google drive](https://drive.google.com/file/d/1_OwuPjnVXNoOjkQ2q0NcmSWxCZJGgFTf/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1ITcHcYL6RCN2C7TP_v6cRg)(code:su94)] | configs/models/tada_r50_ssv2.py | 
| SSV2 | TAda2D | R50 | IN-1K | 2 x 3 | 16 | 65.2 | 89.1 | [[google drive](https://drive.google.com/file/d/1mwINu9ZFUMk1bHt47Xq9CDNnb07PCkG7/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1hXKpTSSpoVDBWpHWhKgjTg)(code:k03n)] | configs/models/tada_r50_ssv2.py | 

## Epic-Kitchens Action Recognition

| architecture | init | resolution | clips x crops | #frames x sampling rate | action acc@1 | verb acc@1 | noun acc@1 | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | 
| ViViT Fact. Enc.-B16x2 | K700 | 320 | 4 x 3 | 32 x 2 | 46.3 | 67.4 | 58.9 | [[google drive](https://drive.google.com/file/d/1ELvwZYeqdsPmDcX1v7_RbFqbQnvHt9sB/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1zOtIAY6neFshmkPR9SuX8g)(code:rinh)] | configs/models/vivit_fac_enc_b16x2_ek100.py |
| ir-CSN-R152 | K700 | 224 | 10 x 3 | 32 x 2 | 44.5 | 68.4 | 55.9 | [[google drive](https://drive.google.com/file/d/1YEIhijzN2aFXyfDL34WB6Q9strYP7WaU/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1swVIBJInQ75dUZKV-OJwlg)(code:s0uj)] | configs/models/ir-csn_r152_ek100.py | 


## MoSI
Note: for the following models, decord 0.4.1 are used rather than the default 0.6.0 for the codebase.

### Pre-train (without finetuning)
| dataset | backbone | checkpoint | config |
| ------- | -------- | ---------- | ------ |
| HMDB51  | R-2D3D-18 | [[google drive](https://drive.google.com/file/d/18wnkUdekhaHGGghjtd77857RA0Ame4oo/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1X3P4jQyuw2AWP-uRgw3YAA)(code:ahqg)]| papers/CVPR2021-MOSI/config/MoSI_r2d3d_hmdb.py |
| HMDB51  | R(2+1)D-10 | [[google drive](https://drive.google.com/file/d/1dbBF0cokI_nCnKaImvXurtYuRQt1jkit/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1K8GyPIkG9KbDnQqi65ObFQ)(code:1ktb)]| papers/CVPR2021-MOSI/config/MoSI_r2p1d_hmdb.py |

### Finetuned
| dataset | backbone | acc@1 | acc@5 | checkpoint | config |
| ------- | -------- | ----- | ----- | ---------- | ------ |
| HMDB51  | R-2D3D-18 | 46.93 | 74.71 | [[google drive](https://drive.google.com/file/d/1A77b3uwxWwlCj0rm7uQcn6m0-uVCUeWQ/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1LfO1fvQ2DD1uoRfS2MH6dA)(code:2puu)]| papers/CVPR2021-MOSI/config/Finetune_r2d3d_hmdb.py | 
| HMDB51  | R(2+1)D-10 | 51.83 | 78.63 | [[google drive](https://drive.google.com/file/d/1OOkooh6_GNsyF_1EolgboN9MFE0O2N2n/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1IhkUv7q7w0JW1ZyuBYgrBA)(code:hgnc)]| papers/CVPR2021-MOSI/config/Finetune_r2p1d_hmdb.py |
