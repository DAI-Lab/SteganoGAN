import os
import json
import shutil
from tqdm import tqdm
from time import time
from glob import glob
from steganogan import SteganoGAN

path_to_images = list(sorted(glob("data/mscoco/val/_/*.jpg")))[:1000]
os.makedirs("models/0", exist_ok=True)
os.makedirs("models/0/images", exist_ok=True)
with open("models/0/metadata.json", "wt") as fout:
    json.dump({
        "cover": True,
        "timestamp": time(),
        "tags": ["MSCOCO"]
    }, fout, indent=2)
for path_to_image in path_to_images:
    shutil.copyfile(path_to_image, "models/0/images/" + os.path.basename(path_to_image))

for path_to_weights in glob("models/**/weights.steg"):
    path_to_imageset = path_to_weights.replace("weights.steg", "imageset")
    os.makedirs(path_to_imageset, exist_ok=True)
    os.makedirs(path_to_imageset + "/images", exist_ok=True)
    
    with open(path_to_imageset + "/metadata.json", "wt") as fout:
        json.dump({
            "cover": False,
            "timestamp": time(),
            "tags": ["MSCOCO", "SteganoGAN"],
            "config": json.load(open(path_to_weights.replace("weights.steg", "config.json"))),
            "log": json.load(open(path_to_weights.replace("weights.steg", "metrics.log"))),
        }, fout, indent=2)

    model = SteganoGAN.load(path_to_weights)
    for cover in tqdm(path_to_images, path_to_weights):
        stega = path_to_imageset + "/images/" + os.path.basename(cover).replace(".jpg", ".png")
        if not os.path.exists(stega):
            model.encode(cover, stega, str(time()))
