from shapely import wkt
from PIL import Image
import pandas as pd
import numpy as np

def crop_image(image, bbox):
    """
    Crop the image based on the bounding box.
    
    Args:
        image (PIL.Image): The image to crop.
        bbox (tuple): A tuple containing (x_min, y_min, x_max, y_max).
    
    Returns:
        PIL.Image: The cropped image.
    """
    return image.crop(bbox)

def paste_image(original_image, new_image, bbox):
    """
    Paste the new image onto the original image at the specified bounding box.
    
    Args:
        original_image (PIL.Image): The original image.
        new_image (PIL.Image): The new image to paste.
        bbox (tuple): A tuple containing (x_min, y_min, x_max, y_max).
    
    Returns:
        PIL.Image: The image with the new image pasted.
    """
    original_image.paste(new_image, (int(bbox[0]), int(bbox[1])))
    return original_image

def transform_image_to_tensor(image):
    # Transform PIL image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image)

def transform_tensor_to_image(tensor):
    # Transform tensor to PIL image
    transform = transforms.Compose([
        transforms.ToPILImage(),
    ])
    return transform(tensor)

def process_images(image_dir, bbox_csv, vae_model):
    """
    Process images by cropping and pasting new images based on bounding boxes.
    
    Args:
        image_dir (str): The directory containing images.
        bbox_csv (str): The path to the CSV file containing bounding boxes.
        vae_model (nn.Module): The VAE model for generating new images.
    
    Returns:
        list: A list of processed images.
    """
    # Load bounding boxes
    bboxes = pd.read_csv(bbox_csv)
    processed_images = []

    for index, row in bboxes.iterrows():
        image_path = row['tile_path']
        image = Image.open(image_path).convert("RGB")
        
        # Extract bounding box coordinates
        polygon = wkt.loads(row['bbox'])
        x_min, y_min, x_max, y_max = polygon.bounds
        
        # Crop the image
        bbox = (x_min, y_min, x_max, y_max)
        cropped_image = crop_image(image, bbox)
        
        # Convert cropped image to tensor and pass to VAE model
        cropped_tensor = transform_image_to_tensor(cropped_image)
        new_tensor = vae_model(cropped_tensor.unsqueeze(0)).squeeze(0)
        new_image = transform_tensor_to_image(new_tensor)
        
        # Paste the new image back onto the original image
        processed_image = paste_image(image.copy(), new_image, bbox)
        processed_images.append(processed_image)

    return processed_images
