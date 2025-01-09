import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# class TrivialAugmentWide:
#     def __init__(self):
#         self.transform = A.Compose([A.NoOp()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

#     def __call__(self, image, bboxes, labels):
#         augmented = self.transform(image=image, bboxes=bboxes, labels=labels)
#         return augmented['image'], augmented['bboxes'], augmented['labels']
    
# class TrivialAugmentWide:
#     def __init__(self):
#         self.transform = T.Compose([T.Lambda(lambda x: x)])  # Identity transformation

#     def __call__(self, image, bboxes, labels):
#         # Apply the identity transformation to the image
#         augmented_image = self.transform(image)
#         # Return the image and bounding boxes as they are
#         return augmented_image, bboxes, labels

class SimpleAugmentWide:
    def __init__(self):
        self.transform = A.Compose([
            A.NoOp(),
            A.Affine(shear={'x': (0.5, 0.5)}),  # ShearX
            A.Affine(shear={'y': (0.5, 0.5)}),  # ShearY
            A.Affine(translate_px=(16, 16)),  # TranslateX
            A.Affine(translate_px=(16, 16)),  # TranslateY
            A.Rotate(limit=(45, 45)),  # Rotate
            # A.RandomBrightnessContrast(brightness_limit=(0.5, 0.5)),  # Brightness
            # A.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),  # Color
            # A.RandomBrightnessContrast(contrast_limit=(0.5, 0.5)),  # Contrast
            # A.Sharpen(alpha=(0.5, 0.5)),  # Sharpness
            # A.Posterize(num_bits=4),  # Posterize
            # A.Solarize(threshold=128),  # Solarize
            # A.Equalize(),  # Equalize
            # A.GridDistortion(num_steps=5, distort_limit=(0.25, 0.25)),  # GridDistortion
            # A.GridDropout(ratio=0.25),  # GridDropout
            # A.ElasticTransform(alpha=0.5, sigma=50),  # ElasticTransform
            # A.CoarseDropout(max_holes=1, max_height=112, max_width=112, min_holes=1, min_height=1, min_width=1),  # RandomErasing
            # A.ShiftScaleRotate(shift_limit=0.25, scale_limit=0.25, rotate_limit=45)  # ShiftScaleRotate
        ], 
        bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def __call__(self, image, bboxes, labels):
        augmented = self.transform(image=image, bboxes=bboxes, labels=labels)
        return augmented['image'], augmented['bboxes'], augmented['labels']

class TrivialAugmentWide:
    def __init__(self, num_magnitude_bins: int = 31):
        self.num_magnitude_bins = num_magnitude_bins

    def _augmentation_space(self, num_bins: int):
        return {
            "Identity": (0.0, False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (torch.linspace(1, 7, num_bins).round().int(), False),
            "Solarize": (torch.linspace(255.0, 0.0, num_bins), False),
            "Equalize": (0.0, False),
            "GridDistortion": (torch.linspace(0.0, 0.5, num_bins), True),
            "GridDropout": (torch.linspace(0.0, 0.5, num_bins), True),
            "ElasticTransform": (torch.linspace(0.0, 1.0, num_bins), True),
            "RandomErasing": (torch.linspace(0.0, 1.0, num_bins), True),
            "ShiftScaleRotate": (torch.linspace(0.0, 0.5, num_bins), True),
        }

    def __call__(self, image, bboxes, labels):
        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = (
            float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item())
            if isinstance(magnitudes, torch.Tensor) and magnitudes.ndim > 0
            else 0.0
        )
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        transform = self._get_transform(op_name, magnitude)
        augmented = transform(image=image, bboxes=bboxes, labels=labels)
        return augmented['image'], augmented['bboxes'], augmented['labels']

    def _get_transform(self, op_name, magnitude):
        if op_name == "Identity":
            return A.Compose([A.NoOp()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "ShearX":
            magnitude = int(magnitude)  # Ensure magnitude is an integer
            return A.Compose([A.Affine(shear={'x': (magnitude, magnitude)})], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "ShearY":
            magnitude = int(magnitude)  # Ensure magnitude is an integer
            return A.Compose([A.Affine(shear={'y': (magnitude, magnitude)})], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "TranslateX":
            magnitude = int(magnitude)  # Ensure magnitude is an integer
            return A.Compose([A.Affine(translate_px=(magnitude, 0))], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "TranslateY":
            magnitude = int(magnitude)  # Ensure magnitude is an integer
            return A.Compose([A.Affine(translate_px=(0, magnitude))], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "Rotate":
            return A.Compose([A.Rotate(limit=(magnitude, magnitude))], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "Brightness":
            return A.Compose([A.RandomBrightnessContrast(brightness_limit=(magnitude, magnitude))], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "Color":
            magnitude = int(magnitude)  # Ensure magnitude is an integer
            return A.Compose([A.ColorJitter(brightness=magnitude, contrast=magnitude, saturation=magnitude, hue=magnitude)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "Contrast":
            return A.Compose([A.RandomBrightnessContrast(contrast_limit=(magnitude, magnitude))], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "Sharpness":
            magnitude = abs(magnitude)  # Ensure magnitude is positive
            if magnitude > 1.0:
                den_nom = torch.randint(len(str(magnitude)), (1,))
                if magnitude > den_nom:
                    magnitude = den_nom / magnitude
                else:
                    magnitude = magnitude / den_nom
            return A.Compose([A.Sharpen(alpha=(magnitude, magnitude))], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "Posterize":
            magnitude = int(magnitude)  # Ensure magnitude is an integer
            magnitude = max(1, min(magnitude, 7))  # Ensure magnitude is within the valid range [1, 7]
            return A.Compose([A.Posterize(num_bits=magnitude)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "Solarize":
            return A.Compose([A.Solarize(threshold=magnitude)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "Equalize":
            return A.Compose([A.Equalize()], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "GridDistortion":
            return A.Compose([A.GridDistortion(num_steps=5, distort_limit=(magnitude, magnitude))], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "GridDropout":
            magnitude = max(0.01, magnitude)  # Ensure magnitude is greater than 0
            return A.Compose([A.GridDropout(ratio=magnitude)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "ElasticTransform":
            magnitude = max(0, magnitude)  # Ensure magnitude is non-negative
            return A.Compose([A.ElasticTransform(alpha=magnitude, sigma=50)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "RandomErasing":
            magnitude = max(0, magnitude)  # Ensure magnitude is non-negative
            max_height = max(1, int(magnitude * 224))
            max_width = max(1, int(magnitude * 224))
            return A.Compose([A.CoarseDropout(max_holes=1, max_height=max_height, max_width=max_width, min_holes=1, min_height=1, min_width=1)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        elif op_name == "ShiftScaleRotate":
            return A.Compose([A.ShiftScaleRotate(shift_limit=magnitude, scale_limit=magnitude, rotate_limit=45)], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))
        else:
            raise ValueError(f"Unknown operation {op_name}")