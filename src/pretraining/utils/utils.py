import random
import copy
import torch
import yaml
import random
import cv2
import numpy as np
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Mix(object):

    def __init__(self, p):
        self.p = p 

    def __call__(self, img):
        if random.random() < self.p:
            return img

        x = np.asarray(img).copy()
        x2 = np.flipud(x)
        x3 = np.fliplr(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                r = random.random()
                if r < 0.33:
                    x[i][j] = x2[i][j]
                elif r < 0.66:
                    x[i][j] = x3[i][j]

        return Image.fromarray(x)

class BackgroundInvariancyAugmentation:
    def __init__(self, p, bg_files):
        self.backgrounds = bg_files
        self.p = p
        self.n_backgrounds = len(bg_files)-1

    def pre_processing(self, image):
        mean = np.mean(image, axis=(0,1))
        broadcast_shape = [1,1]
        broadcast_shape[2-1] = image.shape[2]
        mean = np.reshape(mean, broadcast_shape)
        image = image - mean

        stdDv = np.std(image, axis=(0,1))
        broadcast_shape = [1,1]
        broadcast_shape[2-1] = image.shape[2]
        stdDv = np.reshape(stdDv, broadcast_shape)
        image = image  / (stdDv + 1e-8)

        oMin = 1e-8
        iMin = np.percentile(image, 0.25)

        oMax = 255.0 - 1e-8
        iMax = np.percentile(image, 99.75)

        out = image - iMin
        out *= oMax / (iMax - iMin)
        out[out < oMin] = oMin
        out[out > oMax] = oMax
        return out                    

    def __call__(self, img):
        
        if random.random() < self.p:
            return img
 
        img = np.asarray(img)
        img = np.float64(self.pre_processing(img))
        mask = (2*img[:,:,1] - img[:,:,0] - img[:,:,2]) # - (img[:,:,1] - img[:,:,0])
        mask.clip(0,255)

        value = (np.max(mask) - np.min(mask))/3
        mask[ np.where(mask < value) ] = 0.
        mask[ np.where(mask > 0) ] = 255.

        kernel_dilation = np.ones((6,6), np.uint8)
        kernel_erosion = np.ones((3,3), np.uint8)

        mask = cv2.erode(mask, kernel_erosion, iterations=2)
        mask = cv2.dilate(mask, kernel_dilation, iterations=4)

        patch = np.zeros(img.shape, dtype=np.uint8)
        patch[ np.where(mask != 0) ] = img[ np.where(mask != 0) ]
        random_rotation = random.randint(0,180)
        patch = np.asarray(Image.fromarray(patch).rotate(random_rotation))
 
        new_img = np.asarray(Image.open(self.backgrounds[random.randint(0, self.n_backgrounds - 1)]).convert('RGB')).copy()
        new_img[ np.where( patch!=0) ] = patch[ np.where( patch != 0) ]

        return Image.fromarray(new_img)
       
 
class RandomErasing:
    def __init__(self, p, area):
        self.p = p
        self.area = area

    def __call__(self, img):
        if np.random.random() < self.p:
            return img

        new_img = np.asarray(img)
        S_e = (np.random.random() * self.area + 0.1) * new_img.shape[0] * new_img.shape[1] # random area
        tot = 0

        while tot < S_e:
            y , x = np.random.randint(0, new_img.shape[0]-2) , np.random.randint(0, new_img.shape[1]-2)
            wy, wx = np.random.randint(1, new_img.shape[0] - y) , np.random.randint(1, new_img.shape[1] - x)

            if wy * wx > S_e*2:
                continue

            tot += wy * wx

            random_patch = np.random.rand(wy,wx,3)*255
            new_img[ y : y + wy , x : x + wx , : ] = random_patch
        
        return Image.fromarray(new_img)

class RandomAffine:
    def __init__(self, p, d, t, scale, s, inter):
        self.p = p
        self.degrees = d
        self.translate = t
        self.scale = scale
        self.shear = s
        self.interpolation = inter

    def __call__(self, img):
        if random.random() < self.p:
            return transforms.RandomAffine(degrees = self.degrees,
                                       translate = self.translate,
                                       scale = self.scale,
                                       shear = self.shear,
                                       interpolation = self.interpolation)(img)
        else:
            return img

class Transform:
    def __init__(self):

        # this should have a parameter in cfg passed to it
        with open("E:/PhD/Paper2/dataset/backgrounds.yaml") as f:
            self.bg_files = yaml.safe_load(f)['pre-training']

        self.transform = transforms.Compose([
           
            BackgroundInvariancyAugmentation(0.8, self.bg_files),

            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=[ 0 , 0.125])],
                p=1.0,
            ),

            Mix(p=0.9),

            GaussianBlur(p=0.9),

            RandomErasing(p=1.0, area = 0.35),                
           
            RandomAffine(p = 0.8,
                         d = 180,
                         t = (0.23,0.25),
                         scale = (0.5, 2),
                         s = (0.25 , 0.75 , 0.25 , 0.75 ),
                         inter = transforms.InterpolationMode.BICUBIC),

            transforms.Resize((224,224),interpolation = transforms.InterpolationMode.BICUBIC),
            
            transforms.ToTensor(),
            ])


    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform(x)
        return torch.cat((y1.unsqueeze(0), y2.unsqueeze(0)),0)

