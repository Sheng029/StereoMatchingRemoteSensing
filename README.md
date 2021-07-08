## Stereo Matching of High-resolution Remote Sensing Images

This repository contains the (testing) codes and trained models for the paper "Dual-Scale Matching Network for Disparity Estimation of High-Resolution Remote Sensing Images" by Sheng He, Ruqin Zhou and Wanshou Jiang*, including:
1. The proposed DSMNet and its variants;
2. Re-implementation of DenseMapNet;
3. Re-implementation of StereoNet;
4. Re-implementation of PSMNet.


#### Environment
1. Ubuntu 20.04
2. Python 3.7
3. CUDA 11.2 and cuDNN 8.1.1
4. TensorFlow (2.5.0rc3)

All models were trained using an RTX 3090 GPU.


#### Data
The dataset used in our experiment is the track-2 dataset of US3D in 2019 Data Fusion Contest (can be downloaded at https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019).
