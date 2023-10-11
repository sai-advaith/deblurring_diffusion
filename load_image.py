import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

import argparse

def load_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_image", action="store_true")
    parser.add_argument("--load_image_path", help="image file path saved as npz", default=None)
    parser.add_argument("--load_image_save_path", help="path where image has to be saved", default=None)

    parser.add_argument("--rescale_image", action="store_true")
    parser.add_argument("--rescale_image_path", help="image file path saved as npz", default=None)
    parser.add_argument("--rescale_image_save_path", help="path where image has to be saved", default=None)

    parser.add_argument("--plot_curve", action="store_true")
    parser.add_argument("--log_file_path", help="path of log file", default=None)
    parser.add_argument("--plot_image_save_path", help="path where plot has to be saved", default=None)

    return parser

def load(image_path, image_save_path):
    image_val = np.load(image_path) ['arr_0'][0]

    im = Image.fromarray(image_val)
    im.save(image_save_path)

def rescale(image_path):
    img = Image.open(image_path)
    print(f"size before {img.size}")
    transform = T.Resize((256, 256))


    resized_img = transform(img)
    print(f"size after {resized_img.size}")
    resized_img.save("imagenet_test_resize.jpg")

def plot(file_path):
    file1 = open(file_path, 'r')
    values = file1.readlines()

    scalar_values = []
    for value in values:
        scalar_values.append(np.exp(float(value[:-1])))

    plt.plot(range(len(scalar_values)), scalar_values)
    plt.xlabel('Diffusion Timestep')
    plt.ylabel('Probability')

    plt.title('Probability vs Diffusion Timestep')
    plt.savefig('learning_curve_prob.png')

# Create parser for different utility tasks
util_parser = load_parser()

args = util_parser.parse_args()

# rescale('imagenet.JPEG')
if args.load_image:
    load(args.load_image_path, args.load_image_save_path)

# plot('copy.txt')