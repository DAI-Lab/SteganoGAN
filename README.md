# Deep Steganography
This repository contains the research code for the neural steganography project. The 
client code (compression, error correction, etc.) has been removed and will appear 
in a separate repo.

## Setup
Before you can start training the models, you need to download the datasets. We 
provide a Bash script to automate this process:

```
cd data
bash download.sh
```

This process can take up to 24 hours, depending on your internet speed. Next, you
should make sure you meet all the requirements. If you want GPU support, you should
follow the PyTorch installation instructions at https://pytorch.org before installing
the other dependencies by running:

> pip install -r requirements.txt

Once everything is done installing, you can run `train.py` to reproduce our results.
