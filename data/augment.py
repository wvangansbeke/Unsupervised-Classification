# List of augmentations based on randaugment
import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from torchvision.transforms.transforms import Compose

random_mirror = True

def ShearX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Identity(img, v):
    return img

def TranslateX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateXAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateYAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Solarize(img, v):
    return PIL.ImageOps.solarize(img, v)

def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def augment_list():
    l = [
        (Identity, 0, 1),  
        (AutoContrast, 0, 1),
        (Equalize, 0, 1), 
        (Rotate, -30, 30),
        (Solarize, 0, 256),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.1, 0.1),
        (TranslateX, -0.1, 0.1),
        (TranslateY, -0.1, 0.1),
        (Posterize, 4, 8),
        (ShearY, -0.1, 0.1),
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

class Augment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (random.random()) * float(maxval - minval) + minval
            img = op(img, val)

        return img

def get_augment(name):
    return augment_dict[name]

def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)

class Cutout(object):
    def __init__(self, n_holes, length, random=False):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        length = random.randint(1, self.length)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img
