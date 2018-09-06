# PyTorch DCEC

This repository contains DCEC method ([Deep Clustering with Convolutional Autoencoders](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf)) implementation with PyTorch with some improvements for network architectures.

The code for clustering was developed for Master Thesis: "Automatic analysis of images from camera-traps" by Michal Nazarczuk from Imperial College London

## Prerequisites

The following libraries are required to be installed for the proper code evaluation:

1. PyTorch 
2. NumPy
3. scikit-learn
4. [TensorboardX](https://github.com/lanpa/tensorboardX)

The code was written and tested on Python 3.4.1

## Installation and usage

### Installation

Just copy the repository to your local folder:
```
git clone https://github.com/michaal94/torch_DCEC
```

### Use of the algortihm

In order to test the basic version of the semi-supervised clustering just run it with your python distribution you installed libraries for (Anaconda, Virtualenv, etc.). In general type:

```
cd torch_DCEC
python3 torch_DCEC.py
```
The example will run sample clustering with MNIST-train dataset.

## Options

The algorithm offers a plenty of options for adjustments:
1. Mode choice: full or pretraining only, use:
    ```--mode train_full``` or ```--mode pretrain```
    
    Fot full training you can specify whether to use pretraining phase ```--pretrain True``` or use saved network ```--pretrain False``` and 
    ```--pretrained net ("path" or idx)``` with path or index (see catalog structure) of the pretrained network
2. Dataset choice:
    + MNIST - train, test, full
    + Custom dataset - use the following data structure (characteristic for PyTorch):
        ```
        -data_directory (clusters must corespond to real clustering only for statistics)
            -cluster_1
                -image_1
                -image_2
                -...
            -cluster_2
                -image_1
                -image_2
                -...
            -...
        ```
    Use the following: ```--dataset MNIST-train```, 
    ```--dataset MNIST-test```, 
    ```--dataset MNIST-full``` or 
    ```--dataset custom``` (use the last one with path 
    ```--dataset_path 'path to your dataset'``` 
    and the trasformation you want for images 
    ```--custom_img_size [height, width, depth]```)
3. Different network architectures:
    + CAE 3 - convolutional autoencoder used in [DCEC](https://xifengguo.github.io/papers/ICONIP17-DCEC.pdf) ```--net_architecture CAE_3```
    + CAE 3 BN - version with Batch Normalisation layers ```--net_architecture CAE_3bn```
    + CAE 4 (BN) - convolutional autoencoder with 4 convolutional blocks ```--net_architecture CAE_4``` and ```--net_architecture CAE_4bn```
    + CAE 5 (BN) - convolutional autoencoder with 5 convolutional blocks ```--net_architecture CAE_5``` and ```--net_architecture CAE_5bn``` (used for 128x128 photos)
    
    The following opions may be used for model changes:
    + LeakyReLU or ReLU usage: ```--leaky True/False``` (True provided better results)  
    + Negative slope for Leaky ReLU: ```--neg_slope value``` (Values around 0.01 were used)
    + Use of sigmoid and tanh activations at the end of encoder and decoder: ```--activations True/False``` (False provided better results)
    + Use of bias in layers: ```--bias True/False```
4. Optimiser and scheduler settings (Adam optimiser):
    + Learning rate: ```--rate value``` (0.001 is reasonable value for Adam)
    + Learning rate for pretraining phase: ```--rate_pretrain value``` (0.001 can be used as well)
    + Weight decay: ```--weight value``` (0 was used)
    + Weight decay for pretraining phase: ```--weight_pretrain value```
    + Scheduler step (how many iterations till the rate is changed): ```--sched_step value```
    + Scheduler step for pretraining phase: ```--sched_step_pretrain value```
    + Scheduler gamma (multiplier of learning rate): ```--sched_gamma value```
    + Scheduler gamma for pretraining phase: ```--sched_gamma_pretrain value```
5. Algorithm specific parameters:
    + Clustering loss weight (for reconstruction loss fixed with weight 1): ```--gamma value``` (Value of 0.1 provided good results)
    + Update interval for target distribution (in number of batches between updates): ```update_interval value``` (Value may be chosen such that distribution is updated each 1000-2000 photos)
    + Stop criterium tolerance ```--tol value``` (Depends on dataset, for small 0.01 was used for bigger e.g. MNIST - 0.001)
    + Target number of clusters ```--num_clusters value```
6. Other options:
    + Batch size: ```--batch_size value``` (Depend on your device, but remember that [too much may be bad for convergence](https://towardsdatascience.com/recent-advances-for-a-better-understanding-of-deep-learning-part-i-5ce34d1cc914))
    + Epochs if stop criterium not met: ```--epochs value```
    + Epochs of pretraining: ```--epochs_pretrain value``` (300 epochs were used, 200 with 0.001 lerning rate and 100 with 10 times smaller - ```--sched_step_pretrain 200```, ```--sched_gamma_pretrain 0.1```)
    + Report printing frequency (in batches): ```--printing_frequency value```
    + Tensorboard export: ```--tensorboard True/False```
    
## Catalog structure
    
The code creates the following catalog structure when reporting the statistics:
```
-Reports
    -(net_architecture_name)_(index).txt
-Nets (copies of weights
    -(net_architecture_name)_(index).pt
    -(net_architecture_name)_(index)_pretrained.txt
-Runs
    -(net_architecture_name)_(index)  <- directory containing tensorboard event file
```
The files are indexed automatically for the files not to be accidentally overwritten. 

## See also

For semi-supervised clustering vistit my [other repository](https://github.com/michaal94/Semisupervised-Clustering)