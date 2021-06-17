## Stereo Matching of Remote Sensing

This repository contains the (testing) codes and trained models for the paper "Dual Scale Stereo Matching Network for Disparity Estimation of High-resolution Remote Sensing Images" by Sheng He and Wanshou Jiang*, including:
1. Our proposed network;
2. Re-implementation of DenseMapNet;
3. Re-implementation of StereoNet;
4. Re-implementation of (simplified) PSMNet.


#### Environment
1. Ubuntu 20.04
2. Python 3.7
3. CUDA 11.2 and cuDNN 8.1.1
4. TensorFlow (2.5.0rc3)
All models were trained using an RTX 3090 GPU.


#### Data
The dataset used in our experiment is the track-2 dataset of US3D in 2019 Data Fusion Contest (can be downloaded at https://ieee-dataport.org/open-access/data-fusion-contest-2019-dfc2019). We re-splitted the training set into training, validation, and testing subsets (see the txt files) due to that the official validation and testing sets haven't released the ground-truth labels.


#### Usage
Directly run dssmnet.py, densemapnet.py, stereonet.py, or psmnet.py to obtain estimated disparity maps (modify the arguments if necessary), use show.py for visualization.
