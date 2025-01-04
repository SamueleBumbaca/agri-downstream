import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from shapely import wkt
from data_augmentation.config import load_config

class ImageDataset(Dataset):
    def __init__(self, config, transform=None):
        self.image_dir = config.image_dir
        self.bboxes = pd.read_csv(config.bbox_file)
        self.transform = transform
        self.split_path = config.split
        self.bboxes = self.get_training_samples()

    def __len__(self):
        return len(self.bboxes)

    def __getitem__(self, idx):
        row = self.bboxes.iloc[idx]
        image_path = row['tile_path']
        image = Image.open(image_path).convert("RGB")
        
        # Extract bounding box coordinates
        polygon = wkt.loads(row['bbox'])
        x_min, y_min, x_max, y_max = polygon.bounds
        bbox = (int(x_min), int(y_min), int(x_max), int(y_max))
        
        # Crop the image
        cropped_image = image.crop(bbox)
        
        if self.transform:
            cropped_image = self.transform(cropped_image)
        
        return cropped_image
    
    def get_training_samples(self):
        split = load_config(self.split_path)
        return self.bboxes[self.bboxes['tile_path'].isin(split['train'])]