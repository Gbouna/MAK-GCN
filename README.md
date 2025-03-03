# MAK-GCN
This is the official repository for **Multi-Head Adaptive Graph Convolution Network for Sparse Point Cloud-Based Human Activity Recognition**

# Prerequisites

### Use the following guide to set up the training environment.

```
Create conda environment with python 3.8
Install cuda toolkit using the command below from this link https://anaconda.org/nvidia/cuda-toolkit

conda install nvidia/label/cuda-11.8.0::cuda-toolkit

conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia

conda install conda-forge::tqdm

conda install conda-forge::pytorch_geometric

```
`
# Data Preparation

## Download datasets.
Download the JHMDB and SHREC datasets using the links below:
```
JHMDB raw data download link:   http://jhmdb.is.tue.mpg.de/challenge/JHMDB/datasets
SHREC raw data download link:   http://www-rech.telecom-lille.fr/shrec2017-hand/
```

## Process datasets.
Use the preprocessing code in the data processing folder to process the data and put them in the data folder. 

Or you can get the already processed data directly from [this GitHub repo](https://github.com/fandulu/DD-Net)

A sample of the processed file is currently in the data folder, please replace it. 

# Training

For JHMDB, run `python train.py --batch-size 512 --epochs 600 --dataset 0 --lr 0.001 | tee train.log`

For SHREC coarse, run `python train.py --batch-size 512 --epochs 600 --dataset 1 --lr 0.001 | tee train.log`

For SHREC fine, run `python train.py --batch-size 512 --epochs 600 dataset 2 --lr 0.001 | tee train.log`

# Testing

To test the trained model, bring the saved model to the main directory and pass its name as an arg for the model-path or simply pass the path to where the model was saved

For JHMDB, run `python test.py --model-path model.pt --dataset 0`

For SHREC coarse, run `python test.py --model-path model.pt --dataset 1`

For SHREC fine, run `python test.py --model-path model.pt --dataset 2`

To force the model to be loaded with CPU run `python test.py --model-path model.pt --dataset 0 --no-cuda`

# Action Recognition in Real-time with HT-ConvNet

![Privacy-Centric Activity Recognition](https://github.com/user-attachments/assets/6ee5d4a8-7afb-4aab-a175-29f745f97dd6)


## Check here for full video

[![Privacy-Centric Activity Recognition](https://img.youtube.com/vi/FExfkhTpHJA/0.jpg)](https://www.youtube.com/watch?v=FExfkhTpHJA)



