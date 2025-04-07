# MAK-GCN
This is the official repository for **Multi-Head Adaptive Graph Convolution Network for Sparse Point Cloud-Based Human Activity Recognition**

## Prerequisites

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

### MiliPoint Dataset.

Download the MiliPoint dataset from their [Google drive](https://drive.google.com/file/d/1rq8yyokrNhAGQryx7trpUqKenDnTI6Ky/view) or from [the GitHub repo](https://github.com/yizzfz/MiliPoint). Unzip the downloaded data and put the contents in data/raw/ according to the file structure below.

In the Milipoint folder, according to the file structure below, make a directory data/processed/mmr_action, where the processed data will be stored.

The `data.py` script in the MiliPoint directory prepares the dataset for training as specified by the data providers.

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

### MMActivity Dataset.

Download the MMActivity dataset from their [GitHub repo](https://github.com/nesl/RadHAR/tree/master/Data)

The data consist of two folders: train and test. Each of these folders further contains subfolders corresponding to the respective activity classes.

Then, run the `process.py` script to prepare the data. This will generate pickle files for each action class in the train and test folders. Copy the generated pickle files to the corresponding train and test folders in the data/raw directory, following the file structure below.

In the MMActivity folder, according to the file structure below, make a directory data/processed/mmr_action, where the processed data will be stored.

The `data.py` script in the MMActivity directory prepares the dataset for training as specified by the data providers.

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

# Training and Testing

### MiliPoint Dataset.

First,  go into the MiliPoint directory, then run `python train.py --use_sgd` to train and `python train.py --eval` to test the trained model. 

### MiliPoint Dataset.

First,  go into the MMActivity directory, then run `python train.py --use_sgd` to train and `python train.py --eval` to test the trained model. 
