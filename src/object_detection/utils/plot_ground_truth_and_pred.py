import yaml
import torch
import fiona
import rasterio
from rasterio.mask import mask
from shapely.geometry import shape, Point
from shapely.affinity import affine_transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from object_detection.models.faster_RCNN_FPN import MyFasterRCNN
import click
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import Normalize
from skimage import io
from object_detection.dataset.field_data_module import FieldDataModule

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(cfg, checkpoint_path):
    model = MyFasterRCNN(cfg)
    model.load_checkpoint(checkpoint_path)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model

def get_test_tiles(shp_path, raster_path, output_folder='temp'):
    os.makedirs(output_folder, exist_ok=True)
    with fiona.open(shp_path, 'r') as shapefile:
        shp_crs = shapefile.crs
        with rasterio.open(raster_path) as src:
            raster_crs = src.crs
            if shp_crs != raster_crs:
                raise ValueError("CRS of the shapefile and raster do not match")
            new_raster_paths = []
            geometries = {}
            for feature in shapefile:
                polygon = shape(feature['geometry'])
                name = feature['properties']['NAME']
                out_image, out_transform = mask(src, [polygon], crop=True)
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                output_path = os.path.join(output_folder, name)
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                new_raster_paths.append(output_path)
                geometries[name] = polygon
    return new_raster_paths, geometries

class CustomTileDataset(Dataset):
    def __init__(self, tile_paths, transform=None):
        self.tile_paths = tile_paths
        self.transform = transform

    def __len__(self):
        return len(self.tile_paths)

    def __getitem__(self, idx):
        img_path = self.tile_paths[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, img_path

def predict_bounding_boxes(model, dataloader):
    model.eval()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)
    all_predictions = {}
    for images, path in dataloader:
        tile_name = os.path.basename(path[0])
        images = [img.to(device) for img in images]
        with torch.no_grad():
            predictions = model(images)
        all_predictions[tile_name] = {'tile_path': path, 'image_coords': predictions}
    return all_predictions

def coords_transform(tile_name, geometries, boxes, transform_type='image_to_geo', output_folder='temp'):
    tile_name = os.path.basename(tile_name)
    tile_path = os.path.join(output_folder, tile_name)
    shapely_geometry = geometries[tile_name]
    bounds = shapely_geometry.bounds
    minx, miny, maxx, maxy = bounds
    tile_width = maxx - minx
    tile_height = maxy - miny
    with rasterio.open(tile_path) as src:
        image_height, image_width = src.height, src.width
    scale_x = tile_width / image_width
    scale_y = tile_height / image_height
    transformation_matrix = [scale_x, 0, 0, -scale_y, minx, maxy]

    transformed_boxes = []
    match transform_type:
        case 'image_to_geo':
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                transformed_points = []
                for x, y in points:
                    point = Point(x, y)
                    transformed_point = affine_transform(point, transformation_matrix)
                    transformed_points.append((transformed_point.x, transformed_point.y))
                transformed_boxes.append(transformed_points)
        case 'geo_to_image':
            inverse_transformation_matrix = [1/scale_x, 0, 0, -1/scale_y, -minx/scale_x, maxy/scale_y]
            for box in boxes:
                xmin, ymin, xmax, ymax = box
                points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
                transformed_points = []
                for x, y in points:
                    point = Point(x, y)
                    transformed_point = affine_transform(point, inverse_transformation_matrix)
                    transformed_points.append((transformed_point.x, transformed_point.y))
                transformed_boxes.append(transformed_points)
        case _:
            raise ValueError("Invalid transform_type. Use 'image_to_geo' or 'geo_to_image'.")

    return torch.tensor(transformed_boxes)

def plot_bounding_boxes(dataset, ids_perc, cfg, predictions):
    if dataset == 'train':
        stage = 'fit'
    elif dataset == 'val':
        stage = 'fit'
    elif dataset == 'test':
        stage = 'test'
    else:
        raise ValueError("Dataset must be 'train', 'val', or 'test'")

    with open(cfg, 'r') as file:
        cfg = yaml.safe_load(file)

    data_module = FieldDataModule(cfg)
    data_module.setup(stage=stage)

    if dataset == 'train':
        data = data_module.train_dataloader()
    elif dataset == 'val':
        data = data_module.val_dataloader()
    elif dataset == 'test':
        data = data_module.test_dataloader()
    else:
        raise ValueError("Dataset must be 'train', 'val', or 'test'")

    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=1)

    for id_p in ids_perc:
        idx = int(len(data.dataset) * id_p)
        img, target = data.dataset[idx]
        img = transforms.ToTensor()(img)  # Convert image to tensor
        fig, ax = plt.subplots(1)
        ax.imshow(img.permute(1, 2, 0))
        print(f"ID: {idx}")

        for box in target['boxes']:
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=1, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            print(f"Ground truth bounding box: {box}")

        tile_name = os.path.basename(predictions[idx]['tile_path'][0])
        for box, score in zip(predictions[tile_name]['image_coords'][0]['boxes'], predictions[tile_name]['image_coords'][0]['scores']):
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            color = cmap(norm(score.cpu().numpy()))
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            print(f"Predicted bounding box: {box}, Score: {score}")

        plt.show()

@click.command()
@click.option('-cfg', '--config', required=True, type=str, help='Path to the config file')
@click.option('-ckpt', '--checkpoint', required=True, type=str, help='Path to the checkpoint file')
@click.option('-out', '--output_file', required=True, type=str, help='Path to the output shapefile')
def main(config, checkpoint, output_file):
    config_path = config
    checkpoint_path = checkpoint
    output_file_path = output_file

    print(f'Config Path: {config_path}')
    print(f'Checkpoint Path: {checkpoint_path}')
    print(f'Output File Path: {output_file_path}')

    cfg = load_config(config_path)

    trial_date = cfg['dataset']['trial'] + '_' + cfg['dataset']['date']
    raster = os.path.join(cfg['dataset']['path_to_dataset'], trial_date, 'orthomosaic.tif')

    model = load_model(cfg, checkpoint_path)
    tile_paths, geometries = get_test_tiles('temp/test_tiles.shp', raster)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg['dataset']['image_size'], cfg['dataset']['image_size']))
    ])

    dataset = CustomTileDataset(tile_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print('Predicting bounding boxes...')
    predictions = predict_bounding_boxes(model, dataloader)
    print('Transforming image coordinates to crs coordinates...')
    plot_bounding_boxes('test', [.015, 0.056, .124, .21658, .32156, .489651, .567541, .61354, .785646, .82165464, .912154, .99541], config_path, predictions)

if __name__ == '__main__':
    main()