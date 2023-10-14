import numpy as np
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F

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

    parser.add_argument("--evaluation_metric", action="store_true")
    parser.add_argument("--test_img_path", help="comparision image", default=None)
    parser.add_argument("--reference_img_path", help="reference image", default=None)

    return parser

def load(image_path, image_save_path):
    image_val = np.load(image_path) ['arr_0']

    for i in range(image_val.shape[0]):
        im = Image.fromarray(image_val[i])
        im.save(f'{image_save_path}_{i}.jpg')

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

def transform_img(img_path):
    # Define transformation
    transform = T.Compose([
        T.ToTensor(),
    ])

    img = Image.open(img_path)
    img_tensor = transform(img).cuda() * 255.0
    img_tensor = img_tensor.to(torch.uint8)
    img_tensor = img_tensor.permute(1, 2, 0)

    return img_tensor

def measure_eval(image_path1, image_path2):
    img1_tensor, img2_tensor = transform_img(image_path1), transform_img(image_path2)
 
    # PSNR
    mse = torch.mean((img1_tensor - img2_tensor) ** 2 + 1e-9, dtype=torch.float32)
    psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))

    print(f"PSNR between the {image_path1}, {image_path2} is: {psnr}")
# Create parser for different utility tasks
util_parser = load_parser()

args = util_parser.parse_args()

# rescale('imagenet.JPEG')
if args.load_image:
    load(args.load_image_path, args.load_image_save_path)

if args.evaluation_metric:
    measure_eval(args.test_img_path, args.reference_img_path)

# plot('copy.txt')