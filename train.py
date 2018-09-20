"""
This script trains a steganography model on some dummy data.
"""
import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from scipy import misc
from random import random
from data import yield_images
from model import Steganographer

EPOCHS = 32
DATA_DEPTH = 1
BATCH_SIZE = 8

device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')

model = Steganographer(DATA_DEPTH).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, 'min')

def run(epoch):
    global model
    
    decoding_losses = []
    classifier_losses = []
    iterator = tqdm(yield_images(mode="train"))
    model = model.train()
    for image in iterator:
        _, height, width = image.size()
        image = autograd.Variable(image.to(device).expand(BATCH_SIZE, 3, height, width))
        data = autograd.Variable(torch.zeros((BATCH_SIZE, DATA_DEPTH, height, width)).random_(0, 2).to(device))

        optimizer.zero_grad()
        stegno, decoding_loss, classifier_loss = model(image, data)
        (decoding_loss + classifier_loss).backward()
        optimizer.step()

        decoding_losses.append(decoding_loss.item())
        classifier_losses.append(classifier_loss.item())

        iterator.set_description("TRAIN %.02f, %.02f, %.02f" % (
            decoding_losses[-1],
            sum(decoding_losses) / len(decoding_losses),
            sum(classifier_losses) / len(classifier_losses),
        ))
        if random() < 0.001:
            image = image[0].permute(2,1,0).data.cpu().numpy()*125.0+125.0
            misc.imsave("weights/train.epoch-%s.raw.png" % epoch, np.maximum(0.0, np.minimum(255.0, image)))
            image = stegno[0].permute(2,1,0).data.cpu().numpy()*125.0+125.0
            misc.imsave("weights/train.epoch-%s.stegno.png" % epoch, np.maximum(0.0, np.minimum(255.0, image)))

    decoding_losses = []
    classifier_losses = []
    iterator = tqdm(yield_images(mode="test"))
    model = model.eval()
    for image in iterator:
        _, height, width = image.size()
        image = autograd.Variable(image.to(device).expand(BATCH_SIZE, 3, height, width))
        data = autograd.Variable(torch.zeros((BATCH_SIZE, DATA_DEPTH, height, width)).random_(0, 2).to(device))

        stegno, decoding_loss, classifier_loss = model(image, data)
        decoding_losses.append(decoding_loss.cpu().data[0])
        classifier_losses.append(classifier_loss.cpu().data[0])

        iterator.set_description("TEST: %.02f, %.02f" % (
            sum(decoding_losses) / len(decoding_losses),
            sum(classifier_losses) / len(classifier_losses),
        ))
        if random() < 0.001:
            image = image[0].permute(2,1,0).data.cpu().numpy()*125.0+125.0
            misc.imsave("weights/test.epoch-%s.raw.png" % epoch, np.maximum(0.0, np.minimum(255.0, image)))
            image = stegno[0].permute(2,1,0).data.cpu().numpy()*125.0+125.0
            misc.imsave("weights/test.epoch-%s.stegno.png" % epoch, np.maximum(0.0, np.minimum(255.0, image)))

    print("Epoch %s, Decoder %.02f, Classifier %.02f" % (
        epoch,
        sum(decoding_losses) / len(decoding_losses),
        sum(classifier_losses) / len(classifier_losses),
    ))
    scheduler.step(sum(decoding_losses) / len(decoding_losses))
    torch.save(model, "weights/model.epoch-%s.loss-%.02f.pt" % (epoch, sum(decoding_losses) / len(decoding_losses)))

for epoch in range(EPOCHS):
    run(epoch)
