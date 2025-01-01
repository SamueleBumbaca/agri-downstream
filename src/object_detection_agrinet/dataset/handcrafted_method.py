import os
import sys
import numpy as np
import torch
from pathlib import Path
import yaml
import csv
import rasterio
import rasterio.mask
from shapely.geometry import Point, Polygon, shape, box
from shapely.affinity import affine_transform
import fiona
from fiona.crs import from_epsg
from skimage.color import rgb2hsv
from skimage.measure import label, regionprops
from sklearn.cluster import AgglomerativeClustering
from skimage.measure import LineModelND, ransac

class HandCraftedBBoxExtractor():
    def __init__(self, 
                 directory, 
                 trial, 
                 date,
                 resolution, 
                 min_dist_on_row, 
                 max_dist_on_row,
                 dist_tolerance,
                 inter_row_distance,
                 binary_method,
                 experiment_id,
                 load_orthomosaic=False):
        # experiment parameters
        self.trial = trial
        self.date = date
        self.directory = Path(directory)
        self.experiment_id = experiment_id
        # directories
        trial_date_dir = f"{self.trial}_{self.date}"
        self.folders = self.directory / trial_date_dir
        self.config_path = self.directory / 'split.yaml'
        self.orthomosaic_path = self.folders / 'orthomosaic.tif'
        self.tiles_path = self.folders / 'tiles.shp'
        self.field_shape_path = self.folders / 'field_shape.shp'
        self.base_tiles_path = self.directory / 'base'
        # image and agronomical parameters
        if not isinstance(resolution, float):
            if isinstance(resolution, str) or isinstance(resolution, int):
                try:
                    self.resolution = float(resolution)
                except ValueError:
                    self.resolution = None
            else:
                self.resolution = None
        else:
            self.resolution = resolution
        if load_orthomosaic or not self.resolution:
            print('Loading orthomosaic...')
            self.orthomosaic = self.load_orthomosaic()
        self.min_dist_on_row = float(min_dist_on_row)
        self.max_dist_on_row = float(max_dist_on_row)
        self.dist_tolerance = float(dist_tolerance)
        self.inter_row_distance = float(inter_row_distance) / self.resolution
        self.binary_method = binary_method
        self.min_dist = (self.min_dist_on_row - self.dist_tolerance) / self.resolution
        self.max_dist = (self.max_dist_on_row + self.dist_tolerance) / self.resolution
        # initialize the dataset
        self.test_crs()
        print('Loading tiles...')
        self.tiles, self.shapes = self.load_tiles()
        self.tiles_names = list(self.tiles.keys())
        self.tiles_names_paths = {name: self.base_tiles_path / name for name in self.tiles.keys()}

    def extract_bounding_boxes(self, rewrite=False):
        """
        Extract bounding boxes from the orthomosaic tiles.

        Parameters:
        - rewrite: A boolean indicating whether to rewrite the bounding boxes.

        Returns:
        - bbox_path: The path to the CSV file containing the bounding boxes.
        """
        # Create the directory if it doesn't exist adjust as needed
        output_dir = Path('experiments') / self.experiment_id / 'Handcrafted_dataset'
        os.makedirs(output_dir, exist_ok=True)
        # Check if the bounding boxes have already been extracted
        bbox_path = output_dir / 'bboxes.csv'
        if rewrite or not os.path.exists(bbox_path):
            self.process_observations()
            self.filter_observations_by_slope()
            self.process_objects()
            self.transform_bounding_boxes()
            self.save_bounding_boxes(output_dir, coords_type='img')
        return bbox_path

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        # Get the tile name and geometry
        if isinstance(idx, int):
            tile_name = list(self.tiles.keys())[idx]
        elif isinstance(idx, str):
            tile_name = idx

        tile_path = self.base_tiles_path / tile_name

        # Load the orthomosaic image
        with rasterio.open(tile_path) as src:
            out_image = src.read()
            out_image = np.transpose(out_image, (1, 2, 0))  # Convert to HWC format
        # Check if the patch is square and of the correct size
        if out_image.shape != (224, 224, 4):
            # Get the index from the tile name
            idx = next(i for i, name in enumerate(self.tiles.keys()) if name == tile_name)
            next_idx = (idx + 1) % len(self.tiles)
            return self.__getitem__(next_idx)

        # Convert the patch to a tensor
        patch_tensor = torch.tensor(out_image, dtype=torch.float32).permute(2, 0, 1)

        return patch_tensor

    def test_crs(self):
        with rasterio.open(self.orthomosaic_path) as src:
            orthomosaic_crs = src.crs
        with fiona.open(self.tiles_path, 'r') as shapefile:
            tiles_crs = shapefile.crs
        if orthomosaic_crs != tiles_crs:
            print('CRS of orthomosaic and tiles do not match')
            sys.exit(1)
        else:
            self.crs = orthomosaic_crs

    def load_tiles(self):
        with open(self.config_path, 'r') as file:
            config = yaml.safe_load(file)
            tiles_to_load = [os.path.basename(file) for file in config['pre-training']]
        with fiona.open(self.field_shape_path, 'r') as shapefile:
            field_shape = shape(shapefile[0]['geometry'])
        with fiona.open(self.tiles_path, 'r') as shapefile:
            tiles = {feature['properties']['NAME']: shape(feature['geometry']) for feature in shapefile if feature['properties']['NAME'] in tiles_to_load and shape(feature['geometry']).intersects(field_shape)}
            shapes = {feature['properties']['NAME']: feature for feature in shapefile if feature['properties']['NAME'] in tiles_to_load and shape(feature['geometry']).intersects(field_shape)}
            print(f'Number of tiles: {len(tiles.keys())}')

        return tiles, shapes
    
    def load_orthomosaic(self):
        with rasterio.open(self.orthomosaic_path) as src:
            # self.orthomosaic = src.read()
            self.orthomosaic_transform = src.transform
            self.orthomosaic_bounds = src.bounds
            self.orthomosaic_profile = src.profile
            self.orthomosaic_meta = src.meta
            self.resolution = src.res

    def get_image(self, idx):
        image_data = self.__getitem__(idx).permute(1, 2, 0).numpy()
        if image_data.dtype == np.float32 or image_data.dtype == np.float64:
            image_data = np.clip(image_data / 255.0, 0, 1)
        if image_data.shape[2] == 4:
            image_data = image_data[:, :, :3]
        return image_data

    def coords_transform(self, tile_name, coords, transform_type='image_to_geo'):
        """
        Transforms coordinates between image and geographical coordinates in the CRS stored in self.crs.
        
        Parameters:
        - tile_name: The name of the tile.
        - coords: A list of (x, y) tuples representing coordinates.
        - transform_type: The type of transformation ('image_to_geo' or 'geo_to_image').
        
        Returns:
        - transformed_coords: A list of (x, y) tuples representing transformed coordinates.
        """

        shapely_geometry = shape(self.shapes[tile_name].geometry)
        bounds = shapely_geometry.bounds
        minx, miny, maxx, maxy = bounds
        tile_width = maxx - minx
        tile_height = maxy - miny
        with rasterio.open(self.base_tiles_path / tile_name) as src:
            image_height, image_width = src.height, src.width
        scale_x = tile_width / image_width
        scale_y = tile_height / image_height
        transformation_matrix = [scale_x, 0, 0, -scale_y, minx, maxy]
        
        transformed_coords = []
        match transform_type:
            case 'image_to_geo':
                for x, y in coords:
                    point = Point(x, y)
                    transformed_point = affine_transform(point, transformation_matrix)
                    transformed_coords.append((transformed_point.x, transformed_point.y))
            case 'geo_to_image':
                inverse_transformation_matrix = [1/scale_x, 0, 0, -1/scale_y, -minx/scale_x, maxy/scale_y]
                for x, y in coords:
                    point = Point(x, y)
                    transformed_point = affine_transform(point, inverse_transformation_matrix)
                    transformed_coords.append((transformed_point.x, transformed_point.y))
            case _:
                raise ValueError("Invalid transform_type. Use 'image_to_geo' or 'geo_to_image'.")
        
        return transformed_coords
    
    # Function to propose regions
    def find_plant_centroids(self, i, method = 'HSV'):
        """
        Find plant centroids in an image using a specified method.

        Parameters:
        - i: The index of the image in the dataset.
        - method: The method to use for finding plant centroids.

        Returns:
        - regions: A list of region properties.
        """
        match method:
            case 'HSV':
                return self.find_plant_centroids_HSV(i)
            case _:
                raise ValueError(f"Invalid method: {method}")

    # Custom function to propose regions
    def find_plant_centroids_HSV(self, i, green_hue_min=30, green_hue_max=80):
        """
        Find plant centroids in an image using the HSV color space.

        Parameters:
        - i: The index of the image in the dataset.
        - green_hue_min: The minimum hue value for green color.
        - green_hue_max: The maximum hue value for green color.

        Returns:
        - regions: A list of region properties.
        """
        # Get the image data from the field dataset
        image_data = self.__getitem__(i).permute(1, 2, 0).numpy()
        def convert_hue(hue):
            return hue * 255
        if image_data.dtype == np.float32 or image_data.dtype == np.float64:
            image_data = np.clip(image_data / 255.0, 0, 1)
        if image_data.shape[2] == 4:
            image_data = image_data[:, :, :3]
        image_hsv = rgb2hsv(image_data)
        hue = convert_hue(image_hsv[:, :, 0])
        green_mask = (hue >= green_hue_min) & (hue <= green_hue_max)
        labeled_patches = label(green_mask)
        regions = []
        # Filter regions by area (10 pixels is arbitrary)
        for region in regionprops(labeled_patches):
            if region.area >= 10:
                regions.append(region)
        return regions

    def extract_ransac_line(self, data_points: np.ndarray, max_distance: int):
        """
        Extract a line from the data points using RANSAC.

        Parameters:
        - data_points: An array of data points.
        - max_distance: The maximum distance for a data point to be considered an inlier.

        Returns:
        - results_inliers: An array of inlier points.
        - model_robust: The RANSAC model.
        """
        model_robust, inliers = ransac(data_points, 
                                       LineModelND, 
                                       min_samples=2, 
                                       residual_threshold=max_distance, 
                                       max_trials=1000)
        results_inliers = []
        results_inliers_removed = []
        for i in range(len(data_points)):
            if not inliers[i]:
                results_inliers_removed.append(data_points[i])
                continue
            x = data_points[i][0]
            y = data_points[i][1]
            results_inliers.append((x, y))
        return np.array(results_inliers), model_robust
    
    def agglomerate_regions(self, regions, min_dist):
        """
        Agglomerate regions using AgglomerativeClustering (tree high cut).

        Parameters:
        - regions: A list of region properties.
        - min_dist: The minimum distance for agglomeration.

        Returns:
        - clust_centroids: An array of cluster centroids.
        """
        centroids = np.array([region.centroid for region in regions])
        clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=min_dist)
        clustering.fit(centroids)
        cluster_labels = clustering.labels_
        clust_centroids = np.array([centroids[cluster_labels == i].mean(axis=0) for i in range(cluster_labels.max() + 1)])
        return clust_centroids

    def process_observations(self):
        """
        Thresholding the image and extracting the first handcrafted detection by RANSAC and agronomic knowledge
        we use the color thresholding function to extract the green color from the image.
        then we use the RANSAC algorithm to extract the first handcrafted detection.
        We use the agronomic knowledge to define the minimum and maximum distance between the plants.
        Wwe use the minimun distance between the plants to clusterize the detections; 
        this is for aglomerate the detections that are close to each other (of the same plant).
        Then we use again the minimum distance on row to define RANSAC maximum distance (confidence).
        Having the detections we can test if candidate plants have maximum 2 plants whithin the
        minimum and maximum distance between plants.
        If so we retain the detections and the RANSAC slope, otherwise we discard it.
        """

        self.observations = {}

        for i in range(len(self)):
            try:
                regions = self.find_plant_centroids(i, method='HSV')
                tile_name = list(self.tiles.keys())[i]
                tile_geometry = self.tiles[tile_name]
                if not isinstance(tile_geometry, Polygon):
                    tile_geometry = Polygon(tile_geometry)
                centroids = np.array([region.centroid for region in regions])
                inlier_points, model = self.extract_ransac_line(centroids, self.min_dist)
                clust_cen = self.agglomerate_regions(regions, self.min_dist)
                distances = np.linalg.norm(clust_cen[:, None] - clust_cen, axis=-1)
                valid_distances = (distances > self.min_dist) & (distances < self.max_dist)
                num_valid_distances = valid_distances.sum(axis=1)
                if (num_valid_distances > 2).any():
                    # print('Found a plant with more than 2 valid distances')
                    # print(f"Warning, {i} is not processed")
                    self.observations[tile_name] = {'centroids': {}, 
                                                    'coordinates':{}, 
                                                    'slope': {}}
                    continue
                transformed_points = self.coords_transform(tile_name, 
                                                           clust_cen, 
                                                           transform_type='image_to_geo')
                points = [Point(x, y) for y,x in clust_cen]
                self.observations[tile_name] = {'centroids': points,
                                           'coordinates': transformed_points, 
                                           'slope': model.params[1]}
            except Exception as e:
                print(e)
                print(f"Warning, {i} is not processed")

    def filter_observations_by_slope(self):
        """
        Filter initial handcrafted detection by ransac slope
        as one assumption is flat field and rows on straght line, the slope (row direction) should be 
        the same for all rows. So we filter the detections by Interquartile range (IQR) to remove the outliers.
        """
        slopes = [value['slope'] for value in self.observations.values() if isinstance(value['slope'], np.ndarray)]
        angles = [np.arctan2(slope[1], slope[0]) * (180 / np.pi) for slope in slopes]
        q1 = np.percentile(angles, 25)
        q3 = np.percentile(angles, 75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        for key, value in self.observations.items():
            slope = value['slope']
            if isinstance(slope, np.ndarray):
                angle = np.arctan2(slope[1], slope[0]) * (180 / np.pi)
                if angle < lower_bound or angle > upper_bound:
                    self.observations[key]['centroids'] = {}
                    self.observations[key]['coordinates'] = {}
                    self.observations[key]['slope'] = {}

    def process_objects(self):
        """
        Complation of handcrafted detection
        we detect the possible other rows in the tile if the tile was not filtered out by the previous steps.
        The other row is searched at inter row distance on the right and left of the first row.
        If the other row is found we clusterize the detections of the other row as done with the first row.
        """
        # Initialize the objects dictionary
        self.objects = {}
        # Get the tile names
        keys = list(self.tiles.keys())
        for tile_name in self.observations:
            # Skip tiles that were filtered out
            if self.observations[tile_name]['centroids'] == {}:
                print(f"Skipping {tile_name}")
                continue
            # Process the tile
            print(f"Processing {tile_name}")
            # Get the index of the tile
            try:
                key_index = keys.index(tile_name)
            except ValueError:
                print(f"'{tile_name}' not found in the list.")
                continue
            # Find the plant centroids in the tile
            regions = self.find_plant_centroids(key_index, method='HSV')
            clust_cen = self.agglomerate_regions(regions, self.min_dist)
            # Get the already fitted first row centroids and the RANSAC model
            first_row_cen = self.observations[tile_name]['centroids']
            first_row_cen = np.array([(point.y, point.x) for point in first_row_cen])
            # Re-fit the first row centroids
            inlier_points, model = self.extract_ransac_line(first_row_cen, self.min_dist)
            # Get the angle (radians) - slope (pixels coef) of the first row
            slope = model.params[1][1] / model.params[1][0]
            angle_rad = np.arctan2(model.params[1][1], model.params[1][0])
            # Get the perpendicular vector to the first row
            direction_vector = np.array([np.cos(angle_rad), np.sin(angle_rad)])
            perpendicular_vector = np.array([-direction_vector[1], direction_vector[0]])
            # Get the rows on the right and left of the first row
            line_minus90_origin = model.params[0] + self.inter_row_distance * perpendicular_vector
            line_plus90_origin = model.params[0] - self.inter_row_distance * perpendicular_vector
            intercept_minus90 = line_minus90_origin[1] - slope * line_minus90_origin[0]
            intercept_plus90 = line_plus90_origin[1] - slope * line_plus90_origin[0]
            # Get the points on the right and left of the first row
            Points_minus90 = []
            Points_plus90 = []
            for point in clust_cen:
                x, y = point
                distance_to_minus90 = abs(slope * x - y + intercept_minus90) / np.sqrt(slope**2 + 1)
                distance_to_plus90 = abs(slope * x - y + intercept_plus90) / np.sqrt(slope**2 + 1)
                if distance_to_minus90 <= self.min_dist:
                    Points_minus90.append(point)
                if distance_to_plus90 <= self.min_dist:
                    Points_plus90.append(point)
            Points_minus90 = np.array(Points_minus90)
            Points_plus90 = np.array(Points_plus90)
            # Get the valid distances between candidate plants points
            distances_onMinus90 = np.linalg.norm(Points_minus90[:, None] - Points_minus90, axis=-1)
            valid_distances_onMinus90 = (distances_onMinus90 > self.min_dist) & (distances_onMinus90 < self.max_dist)
            try:
                num_valid_distances_onMinus90 = valid_distances_onMinus90.sum(axis=1)
            except:
                num_valid_distances_onMinus90 = np.zeros(0)
            distances_onPlus90 = np.linalg.norm(Points_plus90[:, None] - Points_plus90, axis=-1)
            valid_distances_onPlus90 = (distances_onPlus90 > self.min_dist) & (distances_onPlus90 < self.max_dist)
            try:
                num_valid_distances_onPlus90 = valid_distances_onPlus90.sum(axis=1)
            except:
                num_valid_distances_onPlus90 = np.zeros(0)
            num_valid_distances = np.concatenate((num_valid_distances_onMinus90, num_valid_distances_onPlus90))
            if (num_valid_distances > 2).any():
                self.objects[tile_name] = {'image_centroids': {}}
                continue
            else:
                if Points_minus90.size == 0 and Points_plus90.size == 0:
                    merged_points = inlier_points
                elif Points_minus90.size == 0:
                    merged_points = np.concatenate((inlier_points, Points_plus90), axis=0)
                elif Points_plus90.size == 0:
                    merged_points = np.concatenate((inlier_points, Points_minus90), axis=0)
                else:
                    merged_points = np.concatenate((inlier_points, Points_minus90, Points_plus90), axis=0)
                # Update the objects dictionary with the points from the other row/rows
                self.objects[tile_name] = {'image_centroids': merged_points}

    def transform_bounding_boxes(self):
        """
        Transform the bounding boxes from image coordinates to geographical coordinates.
        """
        # We define minimum distance between plants as the size of the plant bounding box
        bbox_size = self.min_dist
        # Transform coordinates and create bounding boxes
        for idx,img in enumerate(self):
            # Get the tile name
            try:
                tile_name = list(self.tiles.keys())[idx]
            except IndexError:
                continue
            # Get data from the objects dictionary
            try:
                data = self.objects[tile_name]
            except KeyError:
                continue
            # Get the image data from the field dataset
            image_data = img.to('cpu').detach().numpy().transpose(1, 2, 0)
            # Convert the geometry to a Shapely geometry object
            shapely_geometry = shape(self.shapes[tile_name].geometry)
            # Get the bounds of the geometry
            bounds = shapely_geometry.bounds
            minx, miny, maxx, maxy = bounds
            # Calculate the width and height of the tile
            tile_width = maxx - minx
            tile_height = maxy - miny
            # Get the image dimensions
            image_height, image_width = image_data.shape[:2]
            # Create a transformation matrix based on the bounds and scaling
            scale_x = tile_width / image_width
            scale_y = tile_height / image_height
            transformation_matrix = [scale_x, 0, 0, -scale_y, minx, maxy]
            # Create points with the coordinates from image_centroids
            points = [Point(x, y) for y, x in data['image_centroids']]
            # Create bounding boxes of size min_dist around each centroid in image coordinates
            image_bounding_boxes = []
            coordinates_bounding_boxes = []
            for point in points:
                bbox = box(point.x - bbox_size / 2, point.y - bbox_size / 2,
                        point.x + bbox_size / 2, point.y + bbox_size / 2)
                # Adjust the bounding box if it is outside the image border
                if bbox.bounds[0] < 0:
                    bbox = box(0, bbox.bounds[1], bbox.bounds[2], bbox.bounds[3])
                if bbox.bounds[1] < 0:
                    bbox = box(bbox.bounds[0], 0, bbox.bounds[2], bbox.bounds[3])
                if bbox.bounds[2] > image_width:
                    bbox = box(bbox.bounds[0], bbox.bounds[1], image_width, bbox.bounds[3])
                if bbox.bounds[3] > image_height:
                    bbox = box(bbox.bounds[0], bbox.bounds[1], bbox.bounds[2], image_height)
                image_bounding_boxes.append(bbox)
                # Transform the bounding box to WGS UTM32N coordinates
                transformed_bbox = affine_transform(bbox, transformation_matrix)
                coordinates_bounding_boxes.append(transformed_bbox)
            # Update the objects dictionary with the bounding boxes
            self.objects[tile_name]['image_bounding_boxes'] = image_bounding_boxes
            self.objects[tile_name]['coordinates_bounding_boxes'] = coordinates_bounding_boxes

    def save_bounding_boxes(self, output_dir, coords_type='img'):
        """
        Save the bounding boxes to a dictionary (if img) or shapefile (if geo).

        Parameters:
        - output_filepath: The path to the output shapefile.
        - coords_type: The type of coordinates ('img' or 'geo').
        """
        if coords_type == 'img':
            # Define the output file path
            output_filepath = output_dir / 'bboxes.csv'
            # Save the bounding boxes to a CSV file
            with open(output_filepath, 'w', newline='') as csvfile:
                fieldnames = ['tile_name', 'tile_path', 'bbox']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for tile_name, data in self.objects.items():
                    try:
                        tile_path = self.tiles_names_paths[tile_name]
                        for bbox in data['image_bounding_boxes']:
                            writer.writerow({
                                'tile_name': tile_name,
                                'tile_path': tile_path,
                                'bbox': bbox.wkt  # Write the bbox as a WKT (Well-Known Text) string
                            })
                    except KeyError:
                        continue
            print(f"Bounding boxes saved to {output_filepath}")
        elif coords_type == 'geo':
            # Define the CRS
            crs = from_epsg(32632)
            schema = {
                'geometry': 'Polygon',
                'properties': {'FID': 'int'}
            }
            # Define the output file path
            output_filepath = output_dir / 'bboxes.shp'
            # Write the bounding boxes to a shapefile
            with fiona.open(output_filepath, 'w', driver='ESRI Shapefile', schema=schema, crs=crs) as shp:
                fid = 0
                for tile_name, data in self.objects.items():
                    try:
                        for bbox in data['coordinates_bounding_boxes']:
                            shp.write({
                                'geometry': {
                                    'type': 'Polygon',
                                    'coordinates': [list(bbox.exterior.coords)]
                                },
                                'properties': {'FID': fid}
                            })
                            fid += 1
                    except KeyError:
                        print(f"Skipping {tile_name}")
                        continue
            print(f"Shapefile saved to {output_filepath}")