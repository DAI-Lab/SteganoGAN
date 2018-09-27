# Deep Steganography
This repository contains the research code for the neural steganography project.

## Usage
To use this repository to encode and decode images, install the requirements and run
the following:

```
from deepsteganography import Steganographer

in_path = "tests/resources/flag.jpg"
out_path = "tmp.png"

model = Steganographer()
model.encode(in_path, "Hello!", out_path)

print(model.decode(out_path))
```

## Dev Setup
Before you can start training the models, you need to download the datasets. We 
provide a Bash script to automate this process:

```
cd deepsteganography/data
bash download.sh
```

This process can take up to 24 hours, depending on your internet speed. Next, you
should make sure you meet all the requirements. If you want GPU support, you should
follow the PyTorch installation instructions at https://pytorch.org before installing
the other dependencies by running:

> pip install -r requirements.txt

Once everything is done installing, run the unit tests to make sure things are set 
up properly.

> python -m pytest

Finally, you're ready to make modifications to the model architecture and run:

> python train.py
