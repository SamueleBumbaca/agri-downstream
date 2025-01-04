def save_image(image, path):
    from PIL import Image
    image = (image * 255).astype('uint8')  # Convert to uint8
    img = Image.fromarray(image)
    img.save(path)

def load_image(path):
    from PIL import Image
    import numpy as np
    img = Image.open(path).convert('RGB')
    return np.array(img) / 255.0  # Normalize to [0, 1]

def log_message(message):
    import logging
    logging.basicConfig(level=logging.INFO)
    logging.info(message)

def visualize_images(images, titles=None, cols=3):
    import matplotlib.pyplot as plt
    n = len(images)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 5))
    axes = axes.flatten()
    
    for i in range(n):
        axes[i].imshow(images[i])
        axes[i].axis('off')
        if titles is not None:
            axes[i].set_title(titles[i])
    
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()