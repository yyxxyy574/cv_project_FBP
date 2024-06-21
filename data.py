import os
import numpy as np
from PIL import Image
from skimage import io
from torch.utils.data import Dataset

from config.constants import data_fbp5500

class FBP5500(Dataset):
    def __init__(self, names, scores, transform=None):
        self.names = names
        self.scores = scores.tolist()
        self.transform = transform
        
    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, index):
        image = io.imread(os.path.join(data_fbp5500['dir'], "faces", self.names[index]))
        score = self.scores[index]
        sample = {"image": image, "score": score, "class": round(score) - 1, "filename": self.names[index]}
        
        if self.transform:
            sample["image"] = self.transform(Image.fromarray(sample["image"].astype(np.uint8)))
            
        return sample