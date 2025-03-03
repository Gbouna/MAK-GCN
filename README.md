# MAK-GCN
This is the official repository for **Multi-Head Adaptive Graph Convolution Network for Sparse Point Cloud-Based Human Activity Recognition**

# Prerequisites

### Use the following guide to set up the training environment.

```
Create conda environment with python 3.8

Install cuda toolkit using the command below from this link https://anaconda.org/nvidia/cuda-toolkit
conda install nvidia/label/cuda-11.8.0::cuda-toolkit

Then, install the following:
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install conda-forge::tqdm
conda install conda-forge::pytorch_geometric

```
# Data Preparation

## MiliPoint Dataset.

Download the MiliPoint dataset from their [Google drive](https://drive.google.com/file/d/1rq8yyokrNhAGQryx7trpUqKenDnTI6Ky/view) or from [the Github repo](https://github.com/yizzfz/MiliPoint). Unzip the downloaded data and put the contents in data/raw/ according to the file structure below.

In the Milipoint folder, according to the file structure below, make a directory data/processed/mmr_action, where the processed data will be stored.
```
MiliPoint
└─data
  └─raw
    ├─0.pkl
    ├─1.pkl
    ├─...
  └─processed
    └─mmr_action
```

## MMActivity Dataset.

Download the MMActivity dataset from their [Github repo](https://github.com/nesl/RadHAR/tree/master/Data)

The data consist of two folders: train and test. Each of these folders further contains subfolders corresponding to the respective activity classes.

Then, run the `process.py` script to prepare the data. This will generate pickle files for each action class in the train and test folders. Copy the generated pickle files to the corresponding train and test folders in the data/raw directory, following the file structure below.

In the MMActivity folder, according to the file structure below, make a directory data/processed/mmr_action, where the processed data will be stored.

```
MMActivity
└─data
  └─raw
    └─train
      ├─0.pkl
      ├─1.pkl
      ├─...
    └─test
      ├─0.pkl
      ├─1.pkl
      ├─...
  └─processed
    └─mmr_action
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



