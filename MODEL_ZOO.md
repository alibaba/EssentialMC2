# MODEL ZOO

## Kinetics

| Dataset | architecture | depth | init | clips x crops | #frames x sampling rate | acc@1 | acc@5 | checkpoint | config |
| ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| K400 | TAda2D | R50 | IN-1K | 10 x 3 | 8 x 8 | 76.3 | 92.4 | [[google drive](https://drive.google.com/file/d/1-_Yek3lFYpgahU2Q2ITON4rgqm5Voj2E/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1CzJ82frDeH8P4L55aZdZ-Q)(code:jp7d)] | configs/models/tada_r50.py |
| K400 | TAda2D | R50 | IN-1K | 10 x 3 | 16 x 5 | 76.9 | 92.7 | [[google drive](https://drive.google.com/file/d/1Phj2mSna27Gv46tJzjw4a51_qum1HW6x/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1GQZmB_ZKwq1NjmVooICa2w)(code:buww)] | configs/models/tada_r50.py |
| K400 | ViViT Fact. Enc. | B16x2 | IN-21K | 4 x 3 | 32 x 2 | 79.4 | 94.0 | [[google drive](https://drive.google.com/file/d/1xD4uij9DmZojnl1xuWBa-gwm5hUZxDc7/view?usp=sharing)][[baidu](https://pan.baidu.com/s/1iVjKjEMm-6ymUd15ZNqvXw)(code:1t51)] | configs/models/vivit_fac_enc_b16x2.py |
