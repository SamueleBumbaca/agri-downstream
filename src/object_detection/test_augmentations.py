import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch
from object_detection.dataset.field_data_module import FieldDataModule
import yaml
import skimage.io as io

def plot_image_with_bboxes(image, bboxes, original_image, original_bboxes):
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image.permute(1, 2, 0).cpu().numpy())  # Convert CxHxW to HxWxC for plotting
    for bbox in bboxes:
        rect_0 = patches.Rectangle((bbox[0], bbox[1]), 
                                    bbox[2] - bbox[0], 
                                    bbox[3] - bbox[1], 
                                    linewidth=1, 
                                    edgecolor='r', 
                                    facecolor='none')
        ax[0].add_patch(rect_0)
    ax[1].imshow(original_image)
    for bbox in original_bboxes:
        rect_1 = patches.Rectangle((bbox[0], bbox[1]), 
                                    bbox[2] - bbox[0], 
                                    bbox[3] - bbox[1], 
                                    linewidth=1, 
                                    edgecolor='b', 
                                    facecolor='none')
    
        ax[1].add_patch(rect_1)
    
    plt.show()

def main():
    # Load the configuration
    config_path = 'src/object_detection/config/experiment_config/config_Faster_RCNN_TPN_modHSV_SAGIT22_C_4175_2022_05_27.yaml'
    cfg = yaml.safe_load(open(config_path))

    # Initialize the data module
    data_module = FieldDataModule(cfg)
    data_module.setup(stage='fit')

    # # Get a batch from the train dataloader
    # train_loader = data_module.train_dataloader()
    # batch = next(iter(train_loader))

    train_loader = data_module.train_dataloader()

    while input("Press enter to continue...") != "q":
        batch = next(iter(train_loader))

        images, targets, tile_names, op_names, magnitudes = batch

        # Plot the first image and its bounding boxes
        for image, target, tile_name, op_name, magnitude in zip(images, targets, tile_names, op_names, magnitudes):
            bboxes = target['boxes']
            path = data_module.train_dataset.bboxes[tile_name]['tile_path']
            original_image = io.imread(path)
            original_bboxes = data_module.train_dataset.bboxes[tile_name]['image_bounding_boxes']
            print(f'Operation: {op_name}, Magnitude: {magnitude}')
            plot_image_with_bboxes(image, bboxes, original_image, original_bboxes)

if __name__ == "__main__":
    main()