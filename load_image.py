import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

def load(image_path):
    image_val = np.load(image_path) ['arr_0'][0]

    im = Image.fromarray(image_val)
    im.save("7_256_recover_five_scale.jpg")

def rescale(image_path):
    img = Image.open(image_path)
    print(f"size before {img.size}")
    transform = T.Resize((256, 256))


    resized_img = transform(img)
    print(f"size after {resized_img.size}")
    resized_img.save("imagenet_test_resize.jpg")

# rescale('imagenet.JPEG')
load('samples_1x256x256x3.npz')