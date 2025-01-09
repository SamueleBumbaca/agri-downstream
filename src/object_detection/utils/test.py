import os
import yaml
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models import ResNet50_Weights
from torch.utils.data import DataLoader
from object_detection.dataset.field_data_module import FieldDataModule
from object_detection.models.faster_RCNN_FPN import MyFasterRCNN
import click
import time
import json

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(cfg, checkpoint_path):
    model = MyFasterRCNN(cfg)
    model.load_checkpoint(checkpoint_path)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model

def get_test_dataloader(cfg):
    data_module = FieldDataModule(cfg)
    data_module.setup(stage='test')
    return data_module.test_dataloader()

def format_predictions_and_targets(predictions, targets):
    formatted_predictions = [
        {
            'image_id': i,
            'category_id': label.item(),
            'bbox': box.tolist(),
            'score': score.item()
        }
        for i, pred in enumerate(predictions)
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels'])
    ]

    formatted_targets = [
        {
            'id': idx,  # Add unique ID for each target
            'image_id': i,
            'category_id': label.item(),
            'bbox': box.tolist(),
            'area': (box[2] - box[0]).item() * (box[3] - box[1]).item(),
            'iscrowd': 0
        }
        for idx, (i, target) in enumerate(enumerate(targets))
        for box, label in zip(target['boxes'], target['labels'])
    ]

    return formatted_predictions, formatted_targets

def save_annotations_to_json(predictions, targets, predictions_file, targets_file):
    # Define categories (this should match your dataset's categories)
    categories = [
        {"id": 1, "name": "category_1"},
        # Add more categories as needed
    ]
    
    # Format predictions
    image_ids = list(set([pred['image_id'] for pred in predictions]))
    predictions_formatted = {
        'images': [{'id': img_id} for img_id in image_ids],
        'annotations': predictions,
        'categories': categories
    }
    
    # Format targets
    target_image_ids = list(set([tgt['image_id'] for tgt in targets]))
    targets_formatted = {
        'images': [{'id': img_id} for img_id in target_image_ids],
        'annotations': targets,
        'categories': categories
    }
    
    # Save to JSON files
    with open(predictions_file, 'w') as f:
        json.dump(predictions, f)
    
    with open(targets_file, 'w') as f:
        json.dump(targets_formatted, f)

def evaluate_model(model, dataloader):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    all_predictions = []
    all_targets = []
    start_time = time.time()
    for batch in dataloader:
        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.no_grad():
            predictions = model(images)
        all_predictions.extend(predictions)
        all_targets.extend(targets)
    end_time = time.time()
    print(f'Inference Time: {end_time - start_time:.2f} seconds')
    if len(all_predictions) != len(all_targets):
        raise ValueError('Number of predictions and targets do not match')
    print(f'Number of predictions: {len(all_predictions)}')
    print(f'Time / prediction: {(end_time - start_time) / len(all_predictions):.2f} seconds/prediction')
    predictions, targets = format_predictions_and_targets(all_predictions, all_targets)
    print(f'Time to format predictions and targets: {time.time() - end_time:.2f} seconds')
    return predictions, targets

def calculate_metrics(predictions, targets):
    os.makedirs('temp', exist_ok=True)
    # Save predictions and targets to JSON files
    save_annotations_to_json(predictions, targets, 'temp/predictions.json', 'temp/targets.json')
    # Load the annotations from the JSON files
    coco_gt = COCO('temp/targets.json')
    coco_dt = coco_gt.loadRes('temp/predictions.json')
    
    # Create COCOeval object
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    
    # Run evaluation
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    
    # Extract metrics
    metrics = {
        'mAP': coco_eval.stats[0],  # Average Precision (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
        'mAP_50': coco_eval.stats[1],  # AP @[ IoU=0.50 | area=all | maxDets=100 ]
        'mAP_75': coco_eval.stats[2],  # AP @[ IoU=0.75 | area=all | maxDets=100 ]
        'mAP_small': coco_eval.stats[3],  # AP @[ IoU=0.50:0.95 | area=small | maxDets=100 ]
        'mAP_medium': coco_eval.stats[4],  # AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
        'mAP_large': coco_eval.stats[5],  # AP @[ IoU=0.50:0.95 | area=large | maxDets=100 ]
        'AR_1': coco_eval.stats[6],  # Average Recall (AR) @[ IoU=0.50:0.95 | area=all | maxDets=1 ]
        'AR_10': coco_eval.stats[7],  # AR @[ IoU=0.50:0.95 | area=all | maxDets=10 ]
        'AR_100': coco_eval.stats[8],  # AR @[ IoU=0.50:0.95 | area=all | maxDets=100 ]
        'AR_small': coco_eval.stats[9],  # AR @[ IoU=0.50:0.95 | area=small | maxDets=100 ]
        'AR_medium': coco_eval.stats[10],  # AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]
        'AR_large': coco_eval.stats[11],  # AR @[ IoU=0.50:0.95 | area=large | maxDets=100 ]
    }
    
    return metrics

@click.command()
@click.option('-cfg','--config', required=True, type=str, help='Path to the config file')
@click.option('-ckpt','--checkpoint', required=True, type=str, help='Path to the checkpoint file')
def main(config, checkpoint):
    config_path = config
    checkpoint_path = checkpoint

    print(f'Config Path: {config_path}')
    print(f'Checkpoint Path: {checkpoint_path}')
    
    cfg = load_config(config_path)
    model = load_model(cfg, checkpoint_path)
    test_dataloader = get_test_dataloader(cfg)
    
    predictions, targets = evaluate_model(model, test_dataloader)
    print(f'Number of predictions: {len(predictions)}')
    print(f'Prediction example: {predictions[0]}')
    print(f'Number of targets: {len(targets)}')
    print(f'Target example: {targets[0]}')
    metrics = calculate_metrics(predictions, targets)
    
    print(metrics)

if __name__ == '__main__':
    main()