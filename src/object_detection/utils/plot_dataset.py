import matplotlib.pyplot as plt
import matplotlib.patches as patches
from object_detection.dataset.field_data_module import FieldDataModule
import yaml
import numpy as np

def plot_bounding_boxes(dataset, ids_perc, cfg):

    if dataset == 'train':
        stage = 'fit'
    if dataset == 'val':
        stage = 'fit'
    if dataset == 'test':
        stage = 'test'
    else:
        raise ValueError("Dataset must be 'train', 'val', or 'test'")

    with open(cfg, 'r') as file:
        cfg = yaml.safe_load(file)

    data_module = FieldDataModule(cfg)  # Pass the configuration object

    data_module.setup(stage=stage)

    # Select the dataset
    if dataset == 'train':
        data = data_module.train_dataloader()
    elif dataset == 'val':
        data = data_module.val_dataloader()
    elif dataset == 'test':
        data = data_module.test_dataloader()
    else:
        raise ValueError("Dataset must be 'train', 'val', or 'test'")

    # Plot images with bounding boxes
    for id_p in ids_perc:
        idx = int(len(data.dataset)*id_p)
        img, target = data.dataset[idx]
        fig, ax = plt.subplots(1)
        if isinstance(img, np.ndarray):
            ax.imshow(np.transpose(img, (0, 1, 2)))  # Assuming img is a numpy array in CxHxW format
        else:
            ax.imshow(img.permute(1, 2, 0).cpu().numpy())  # Assuming img is a tensor in CxHxW format
        print(f"ID: {idx}")
        for box in target['boxes']:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            
            print(f"Bounding box: {box}")

        plt.show()


if __name__ == "__main__":
    # Example usage
    dataset = 'test'  # 'train', 'val', or 'test'
    ids_perc = [.015, 0.056, .124, .21658, .32156, .489651,
                .567541, .61354, .785646, .82165464, .912154, .99541]  # List of IDs to plot
    cfg = 'src/object_detection/config/experiment_config/config_Faster_RCNN_TPN_modHSV_SAGIT22_C_4175_2022_05_27.yaml'  # Define or import your configuration object here
    plot_bounding_boxes(dataset, ids_perc, cfg)