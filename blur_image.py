import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import math
import argparse

def create_parser():
    parser = argparse.ArgumentParser()

    # Type of blur kernel and size
    parser.add_argument("--gaussian", action="store_true")
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--kernel_size", type=int, help="size of kernel to convolve")

    # Image paths
    parser.add_argument("--load_img_path", help="image path to load image")
    parser.add_argument("--save_img_path", help="image path to save image")

    # Sigma
    parser.add_argument("--sigma", type=float, help="standard deviation for guaussian blur")

def uniform_blur_filter(input_image, kernel_size):
    kernel = torch.ones(1, 1, kernel_size, kernel_size)
    kernel = kernel / (kernel_size * kernel_size)

    channels = input_image.size() [0]

    # Reshape to 2d depthwise convolutional weight
    kernel = kernel.view(1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.cuda()

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=kernel_size, groups=channels, bias=False,
                                      padding=kernel_size // 2)

    gaussian_filter.weight.data = kernel

    gaussian_filter.weight.requires_grad = False
    output_image = gaussian_filter(input_image)
    return output_image

    # measurement_noise = torch.normal(0, 10, size=blurred_image.size())
    # measurement_noise = measurement_noise.cuda()

    # blurred_image = blurred_image + measurement_noise
    # return blurred_image

def gaussian_blur_filter(input_image, kernel_size, sigma):
    # Set these to whatever you want for your gaussian filter
    kernel_size = 5
    sigma = 20

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    mean = (kernel_size - 1)/2.
    variance = sigma**2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    channels = input_image.size() [0]

    gaussian_kernel = (1./(2.*math.pi*variance)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean)**2., dim=-1) /\
                        (2*variance)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_kernel = gaussian_kernel.cuda()
    print(gaussian_kernel)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=kernel_size, groups=channels, bias=False,
                                      padding=(kernel_size - 1) // 2, padding_mode='reflect')

    gaussian_filter.weight.data = gaussian_kernel

    gaussian_filter.weight.requires_grad = False
    output_image = gaussian_filter(input_image)

    # measurement_noise = torch.normal(0, 0.02, size=output_image.size())
    # measurement_noise = measurement_noise.cuda()

    # output_image = output_image + measurement_noise
    # output_image = output_image.clamp(0.0, 1.0)

    return output_image

# Create parser for different blur tasks
blur_parser = create_parser()
args = blur_parser.parser_args()

# Load image
img = Image.open(args.load_img_path)

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Apply transformation
img_tensor = transform(img).cuda()


# Check pixel value range
print(torch.min(img_tensor), torch.max(img_tensor))

kernel_size = args.kernel_size
if args.gaussian:
    noisy_tensor = gaussian_blur_filter(img_tensor, kernel_size, args.sigma)
if args.uniform:
    noisy_tensor = uniform_blur_filter(img_tensor, kernel_size)

print(noisy_tensor.size())
noisy_tensor = noisy_tensor.permute(1, 2, 0)
noisy_tensor = noisy_tensor.cpu().numpy() * 255
noisy_tensor = noisy_tensor.astype(np.uint8)

print(noisy_tensor.shape)

im = Image.fromarray(noisy_tensor)
im.save(args.save_img_path)

img_copy = Image.open(args.save_img_path)
img_copy_tensor = transform(img_copy).cuda()
print(img_copy_tensor.size())