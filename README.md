# Deep Steganography
This repository contains experimental code for the steganography project. At a 
high level, the idea is to use adversarial training to embed arbitrary data 
into images without producing visible artifacts.

## setup
```
conda create -n pytorch python=3.6 numpy scipy
source activate pytorch
conda install pytorch torchvision -c pytorch
```
