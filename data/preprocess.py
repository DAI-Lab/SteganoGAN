import torch
from glob import glob
from tqdm import tqdm
from scipy import misc

cached = []
for path_to_jpg in tqdm(glob("data/caltech256/**/*.jpg")):
    image = misc.imread(path_to_jpg, mode="RGB")/125.0-1.0
    if len(image.shape) != 3:
        continue
    height, width, _ = image.shape
    if height < 16 or width < 16:
        continue
    if height > 480 or width > 480:
        continue
    image = torch.FloatTensor(image).permute(2,1,0)
    cached.append(image)
torch.save(cached, "data/caltech256.pt")
