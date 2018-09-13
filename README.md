# Deep Steganography
This repository contains experimental code for the steganography project. At a 
high level, the idea is to use adversarial training to embed arbitrary data 
into images without producing visible artifacts.

## Setup
```
conda create -n pytorch python=3.6 numpy scipy
source activate pytorch
conda install pytorch torchvision -c pytorch
```

## Usage
To embed a message into a natural image, run the following:

> python demo.py encode --data "Hello World!" --input demo/kevin.jpg --output demo/output.png

To retrieve the message, run this:

> python demo.py decode --output demo/output.png
