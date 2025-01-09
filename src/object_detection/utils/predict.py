import yaml
import torch
import fiona
import rasterio
from rasterio.mask import mask
from shapely.geometry import box as shapely_box, mapping, shape, Point
from shapely.affinity import affine_transform
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from object_detection.models.faster_RCNN_FPN import MyFasterRCNN
import click
import os
# import glob
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import Normalize
from skimage import io

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def load_model(cfg, checkpoint_path):
    model = MyFasterRCNN(cfg)
    model.load_checkpoint(checkpoint_path)
    model = model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model

# def get_test_tiles(cfg):
    # with open(cfg['data']['split_file'], 'r') as file:
    #     split_data = yaml.safe_load(file)
    # return split_data['test']
    # base = os.path.join(cfg['dataset']['path_to_dataset'], 'base')
    # test = glob.glob(os.path.join(base, 'SAGIT22_C_4175_2022_05_27*.tif'))
    # return test

def get_test_tiles(shp_path, raster_path, output_folder = 'temp'):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    # Open the shapefile
    with fiona.open(shp_path, 'r') as shapefile:
        # Get the CRS of the shapefile
        shp_crs = shapefile.crs
        # Open the raster file
        with rasterio.open(raster_path) as src:
            # Get the CRS of the raster
            raster_crs = src.crs
            # Check if the CRS match
            if shp_crs != raster_crs:
                raise ValueError("CRS of the shapefile and raster do not match")
            # List to store the paths of the new rasters
            new_raster_paths = []
            # Iterate over each polygon in the shapefile
            geometries = {}
            for feature in shapefile:
                polygon = shape(feature['geometry'])
                name = feature['properties']['NAME']
                # Crop the raster with the polygon
                out_image, out_transform = mask(src, [polygon], crop=True)
                out_meta = src.meta.copy()
                # Update the metadata
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
                # Save the cropped raster
                output_path = os.path.join(output_folder, name)
                with rasterio.open(output_path, "w", **out_meta) as dest:
                    dest.write(out_image)
                # Add the path to the list
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
        all_predictions[tile_name] = {'tile_path' : path,
                             'image_coords' : predictions}
    return all_predictions

def coords_transform(tile_name, geometries, boxes, transform_type='image_to_geo', output_folder = 'temp'):
    """
    Transforms coordinates between image and geographical coordinates in the CRS stored in self.crs.
    
    Parameters:
    - tile_name: The name of the tile.
    - coords: A list of (x, y) tuples representing coordinates.
    - transform_type: The type of transformation ('image_to_geo' or 'geo_to_image').
    
    Returns:
    - transformed_coords: A list of (x, y) tuples representing transformed coordinates.
    """
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

def create_shapefile(predictions, output_file_path, geometries, raster): #TODO: it does not work
    with rasterio.open(raster) as src:
        crs = src.crs
    schema = {
        'geometry': 'Polygon',
        'properties': {'score': 'float', 
                        'tile_name': 'str',
                        'bbox' : 'str'}
    }
    
    with fiona.open(output_file_path, 'w', driver='ESRI Shapefile', crs=crs, schema=schema) as shp:
        for tile_name, pred in predictions.items():
            boxes = pred['image_coords'][0]['boxes'].cpu().numpy()  # Convert tensor to numpy array on CPU
            scores = pred['image_coords'][0]['scores']
            crs_coords = []
            image_bbox = []
            for box in boxes:
                minx, miny, maxx, maxy = box
                coords = [(minx, miny), (maxx, maxy)]
                image_bbox.append(f'{minx:.2f},{miny:.2f},{maxx:.2f},{maxy:.2f}')
                geo_coords = coords_transform(tile_name, geometries, boxes, transform_type='image_to_geo')
                crs_coords.append(geo_coords)
            for geo_coords, score, imgb in zip(crs_coords, scores, image_bbox):
                minx, miny = geo_coords[0]
                maxx, maxy = geo_coords[1]
                geom = shapely_box(minx, miny, maxx, maxy)
                shp.write({
                    'geometry': mapping(geom),
                    'properties': {'score': score.item(), 
                                   'tile_name': tile_name,
                                   'bbox': imgb}
                })

def plot_bboxes_on_images(predictions, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Create a colormap and normalization
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=0, vmax=1)  # Assuming scores are between 0 and 1

    for tile_name, prediction in predictions.items():
        # Ensure the tile_name does not have double extension
        tile_name = tile_name.replace('.tif.tif', '.tif')
        
        # Find the corresponding image path
        image_path = prediction['tile_path'][0]
        if not os.path.exists(image_path):
            print(f"Image for {tile_name} not found.")
            continue
        
        # Load the image
        image = io.imread(image_path)
        
        # Create a figure and axis
        fig, ax = plt.subplots(1, figsize=(12, 12))
        
        # Display the image
        ax.imshow(image)
        
        # Plot the bounding boxes
        for box, score in zip(prediction['image_coords'][0]['boxes'], prediction['image_coords'][0]['scores']):
            # Move the box coordinates to CPU and convert to numpy
            xmin, ymin, xmax, ymax = box.cpu().numpy()
            color = cmap(norm(score.cpu().numpy()))
            rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
        
        # Add color bar
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Score')
        
        # Save the plot
        output_path = os.path.join(output_folder, f"{tile_name}.png")
        plt.savefig(output_path)
        plt.close(fig)
        print(f"Saved plot for {tile_name} to {output_path}")


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
    raster = os.path.join(cfg['dataset']['path_to_dataset'], 
                            trial_date, 
                            'orthomosaic.tif')

    model = load_model(cfg, checkpoint_path)
    tile_paths, geometries = get_test_tiles('temp/test_tiles.shp', 
                                raster)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((cfg['dataset']['image_size'], cfg['dataset']['image_size']))
    ])

    dataset = CustomTileDataset(tile_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print('Predicting bounding boxes...')
    predictions = predict_bounding_boxes(model, dataloader)
    print('Transforming image coordinates to crs coordinates...')
    #create_shapefile(predictions, output_file_path, geometries, raster)
    #print(f'Shapefile saved to {output_file_path}')
    plot_bboxes_on_images(predictions, output_folder='test')
    print(f'Plots saved to {'test'}')

if __name__ == '__main__':
    main()