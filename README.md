[![PyPI Shield](https://img.shields.io/pypi/v/steganogan.svg)](https://pypi.python.org/pypi/steganogan)
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/steganogan.svg?branch=master)](https://travis-ci.org/DAI-Lab/steganogan)

# SteganoGAN

Steganography tool based on DeepLearning GANs

- Free software: MIT license
- Documentation: https://DAI-Lab.github.io/steganogan
- Homepage: https://github.com/DAI-Lab/steganogan

## Usage
To use this repository to encode and decode images, install the package by 
running `pip install .` and try running the following:

```
from steganogan import Steganographer

in_path = "tests/resources/flag.jpg"
out_path = "tmp.png"

model = Steganographer()
model.encode(in_path, "Hello!", out_path)

print(model.decode(out_path))
```

## Research
Before you can start training the models, you need to download the datasets. We 
provide a Bash script to automate this process:

```
cd research/data
bash download.sh
```

This process can take up to 24 hours, depending on your internet speed. Next, you
should make sure you meet all the requirements. If you want GPU support, you should
follow the PyTorch installation instructions at https://pytorch.org before installing
the other dependencies by running:

> make install-develop

You can start running experiments now!
