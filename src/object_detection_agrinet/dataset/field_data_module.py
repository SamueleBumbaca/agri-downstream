import os
import random
import csv
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from pytorch_lightning import LightningDataModule
from shapely import wkt
from skimage import io
from dataset.handcrafted_method import HandCraftedBBoxExtractor

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)

class FieldDataModule(LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        # Load the configuration
        self.cfg = cfg
        # Load the batch size and number of workers
        self.batch_size = self.cfg['data']['batch_size_train']
        self.num_workers = self.cfg['data']['num_workers']
        # Initialize the maximum number of bounding boxes and the bounding boxes dictionary
        self.max_bboxes = 0
        self.bboxes = {}
        # Initialize the bbox extractor
        self.bbox_extractor = HandCraftedBBoxExtractor(
                trial=self.cfg['dataset']['trial'],
                date=self.cfg['dataset']['date'],
                directory=self.cfg['dataset']['path_to_dataset'],
                resolution=self.cfg['dataset']['resolution'],
                min_dist_on_row=self.cfg['dataset']['min_dist_on_row'],
                max_dist_on_row=self.cfg['dataset']['max_dist_on_row'],
                dist_tolerance=self.cfg['dataset']['dist_tolerance'],
                inter_row_distance=self.cfg['dataset']['inter_row_distance'],
                binary_method=self.cfg['dataset']['binarization_method'],
                experiment_id=self.cfg['experiment']['id'],
                load_orthomosaic=self.cfg['dataset']['load_orthomosaic']
            )
        # Create the split if it does not exist
        if self.cfg['dataset']['bboxes_path'] is None:
            bbox_path = self.bbox_extractor.extract_bounding_boxes()
            self.cfg['dataset']['bboxes_path'] = bbox_path
            # Save the updated config
            with open(self.cfg['config_path'], 'w') as file:
                yaml.dump(self.cfg, file)
        else:
            bbox_path = self.cfg['dataset']['bboxes_path']
        self.create_split(bbox_path, self.cfg['data']['split'])
        # Define the dataset here
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    # Create the split file
    def create_split(self, bbox_path, split_path, remove_existing=True):
        """
        Create a split file from the bounding box file and images paths.
        There is also a sanity check to ensure that the bounding box file and the images paths are consistent.
        
        Args:
            bbox_path (str): Path to the bounding box file.
            split_path (str): Path to the split file.
            remove_existing (bool): Whether to remove existing split files.

        Returns:
            None
        """
        if not os.path.exists(split_path) or remove_existing:
            with open(bbox_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                tiles = []
                for row in reader:
                    tile_name = row['tile_name']
                    tile_path = row['tile_path']
                    # Extract the bounding box from the WKT format
                    polygon = wkt.loads(row['bbox'])
                    bbox = np.array(polygon.bounds, dtype=np.float32)
                    # Check if the bounding box is valid
                    #if len(bbox) == 4:
                    # Add the tile name to the list
                    tiles.append(tile_name)
                    # Add the bounding box to the dictionary
                    bbox = np.array(polygon.bounds, dtype=np.float32)
                    if tile_name not in self.bboxes:
                        self.bboxes[tile_name] = {'tile_path': tile_path, 
                                                    'image_bounding_boxes': []}
                    self.bboxes[tile_name]['image_bounding_boxes'].append(bbox)
                    # Update the maximum number of bounding boxes
                    self.max_bboxes = max(self.max_bboxes, 
                                        len(self.bboxes[tile_name]['image_bounding_boxes']))
            # Shuffle the tiles
            num_tiles = len(tiles)
            indices = list(range(num_tiles))
            random.shuffle(indices)
            # Split the tiles
            train_split = int(self.cfg['data']['train_split'] * num_tiles)
            val_split = train_split + int(self.cfg['data']['val_split'] * num_tiles)
            split = {
                'train': [tiles[i] for i in indices[:train_split]],
                'val': [tiles[i] for i in indices[train_split:val_split]],
                'test': [tiles[i] for i in indices[val_split:]]
            }
            # Ensure the directory exists
            os.makedirs(os.path.dirname(split_path), exist_ok=True)
            with open(split_path, 'w') as file:
                yaml.dump(split, file)
            print(f"Split file created at {split_path}")

    def setup(self, stage=None):
        # Load the split
        with open(self.cfg['data']['split'], 'r') as file:
            split = yaml.safe_load(file)
        # Create the datasets
        if stage == 'fit' or stage is None:
            self.train_dataset = CustomDataset(split['train'], 
                                               self.bboxes, 
                                               max_bboxes=self.max_bboxes)
            self.val_dataset = CustomDataset(split['val'], 
                                             self.bboxes, 
                                             max_bboxes=self.max_bboxes)
        if stage == 'test' or stage is None:
            self.test_dataset = CustomDataset(split['test'], 
                                              self.bboxes, 
                                              max_bboxes=self.max_bboxes)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, 
                          batch_size=self.batch_size, 
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=True,
                          collate_fn=collate_fn)
    def val_dataloader(self):
        return DataLoader(self.val_dataset, 
                            batch_size=self.batch_size, 
                            shuffle=False, 
                            num_workers=self.num_workers,
                            persistent_workers=True,
                            collate_fn=collate_fn)
    def test_dataloader(self):
        return DataLoader(self.test_dataset, 
                            batch_size=self.batch_size, 
                            shuffle=False, 
                            num_workers=self.num_workers,
                            persistent_workers=True,
                            collate_fn=collate_fn)
    
class CustomDataset(Dataset):
    """
    Custom dataset class for the field data.
    """
    def __init__(self,
                 tile_names, 
                 bboxes, 
                 image_size=(224, 224), 
                 max_bboxes=100):
        self.tile_names = tile_names
        self.bboxes = bboxes
        self.image_size = image_size
        self.max_bboxes = max_bboxes
        self.transforms = T.Compose([
            T.ToTensor(),
            T.Resize(image_size)
        ])
    # Define the length of the dataset
    def __len__(self):
        return len(self.tile_names)
    # Define the get item method
    def __getitem__(self, idx):
        tile_name = self.tile_names[idx]
        tile_path = self.bboxes[tile_name]['tile_path']
        # Load the image using skimage
        image = io.imread(tile_path)
        if image.shape[2] == 4:  # Check if there is an alpha channel
            image = image[:, :, :3]  # Remove the alpha channel
        # Apply the transforms
        image = self.transforms(image)
        # Load the bounding boxes
        bbox = self.bboxes[tile_name]['image_bounding_boxes']
        # # Pad bounding boxes to ensure they have the same length
        # padded_bboxes = np.full((self.max_bboxes, 4), -1, dtype=np.float32)
        # num_bboxes = min(len(bbox), self.max_bboxes)
        # if num_bboxes > 0:
        #     padded_bboxes[:num_bboxes] = np.array(bbox[:num_bboxes])
        # # Ensure bounding boxes have the correct shape
        # bbox = [b for b in padded_bboxes if len(b) == 4]
        # Ensure bounding boxes have the correct shape
        bbox = [b for b in bbox if len(b) == 4]
        bbox = np.array(bbox, dtype=np.float32)
        # Create target dictionary
        target = {
            'boxes': torch.tensor(bbox, dtype=torch.float32),
            'labels': torch.ones((len(bbox),), dtype=torch.int64)  # Assuming all objects are of class 1
        }
        return image, target
