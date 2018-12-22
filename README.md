<p align="left">
<img width=15% src="https://dai.lids.mit.edu/wp-content/uploads/2018/06/Logo_DAI_highres.png" alt=“SteganoGAN” />
<i>An open source project from Data to AI Lab at MIT.</i>
</p>

[![PyPI Shield](https://img.shields.io/pypi/v/steganogan.svg)](https://pypi.python.org/pypi/steganogan)
[![Travis CI Shield](https://travis-ci.org/DAI-Lab/SteganoGAN.svg?branch=master)](https://travis-ci.org/DAI-Lab/SteganoGAN)

# SteganoGAN

- License: MIT
- Documentation: https://DAI-Lab.github.io/SteganoGAN
- Homepage: https://github.com/DAI-Lab/SteganoGAN

# Overview

**SteganoGAN** is a steganographic tool that uses adversarial training to hide messages in images.

# Installation

The simplest and recommended way to install SteganoGAN is using `pip`:

```bash
pip install steganogan
```

Alternatively, clone the repository and install it from source running the `make install` command.

```bash
git clone git@github.com:DAI-Lab/SteganoGAN.git
cd SteganoGAN
make install
```

For development, you can use the `make install-develop` command instead in order to install all
the required dependencies for testing, code linting and notebook running.

# Usage

## Command Line

**SteganoGAN** includes a command line interface, which allows to easily hide messages in images
and later on read them back.

### Hide a message inside an image

To encode an image, after **SteganoGAN** has been installed, just execute `steganogan encode`
passing the path to the image to be used as cover and the message to hide in it:

```
steganogan encode [options] path/to/cover/image.png "Message to hide"
```

### Read a message from an image

To decode a message from a generated image, execute `steganogan decode` passing the path
to the image:

```
steganogan decode [options] path/to/generated/image.png
```

### Additional options

The script has some additional options to control its behavior:

* `-o, --output PATH`: Path where the generated image will be stored. Defaults to `output.png`.
* `-a, --architecture ARCH`: Architecture to use, basic or dense. Defaults to dense.
* `-v, --verbose`: Be verbose.
* `--cpu`: force CPU usage even if CUDA is available. This might be needed if there is a GPU
  available in the system but the VRAM amount is too low.

**NOTE**: Make sure to use the same architecture for both encoding and decoding, otherwise
SteganoGAN won't be able to decode the message.

## Python

The main way to interact with **SteganoGAN** from Python is through the class
`steganogan.SteganoGAN`.

This class can be loaded by giving it the path to a pretrained model:

```
>>> from steganogan import SteganoGAN
>>> steganogan = SteganoGAN.load('research/models/dense.steg')
Using CUDA device
```

Once we have loaded our model, we are ready to give it an input image path, the path of the
image that we want to generate, and the message that we want to hide:

```
>>> steganogan.encode('research/input.png', 'research/output.png', 'This is a super secret message!')
Encoding completed.
```

This will generate an `output.png` image that will look almost like the input one and will
contain the message hidden inside it.

After this, when we want to extract the message from the image, we can simply pass it to the
`decode` method:

```
>>> steganogan.decode('research/output.png')
'This is a super secret message!'
```

# Fitting a new model

A usage example notebook has been included in the `research` folder with a step by step guide
about how to fit a new model on a new images dataset, save it to a file, and later on reload it
and use it to encode and decode messages.

A convenience script has been also included in the `research/data` folder to download a couple
of demo datasets to train models with.
