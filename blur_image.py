import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import random

import math
import argparse

random.seed(42)

# Parser to configure the type of kernel
def create_parser():
    parser = argparse.ArgumentParser()

    # Type of blur kernel and size
    parser.add_argument("--gaussian", action="store_true")
    parser.add_argument("--uniform", action="store_true")
    parser.add_argument("--motion", action="store_true")
    parser.add_argument("--multi", action="store_true")
    parser.add_argument("--no_blur", action="store_true")

    # Kernel size
    parser.add_argument("--kernel_size", type=int, help="size of kernel to convolve")

    # Image paths
    parser.add_argument("--load_img_path", help="image path to load image")
    parser.add_argument("--save_img_path", help="image path to save image")

    # Sigma
    parser.add_argument("--sigma", type=float, help="standard deviation for guaussian blur")
    parser.add_argument("--sigma1", type=float, help="standard deviations for multi blur kernels")
    parser.add_argument("--sigma2", type=float, help="standard deviations for multi blur kernels")

    # Batch blurring
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--batch_path", help="batch of images generated path")
    return parser

# Delta function as the kernel
def no_blur_kernel(input_image, kernel_size):
    channels = input_image.size() [0]

    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float32)
    kernel[kernel_size // 2, kernel_size // 2] = 1.0

    kernel_np = (kernel * 255).clamp(0, 255).to(torch.uint8).cpu().numpy()

    # Reshape to 2d depthwise convolutional weight
    kernel = kernel.view(1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.cuda()

    # Create pytorch object
    no_blur_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                     kernel_size=(kernel_size, kernel_size),
                                     groups=channels, bias=False, padding=0,
                                     stride=1, device=input_image.device)

    # Assign previous kernel as the weights for the object
    no_blur_filter.weight.data = kernel
    no_blur_filter.weight.requires_grad = False

    no_blur_image = no_blur_filter(input_image)

    kernel = kernel[:, 0, :, :].permute(1, 2, 0)
    kernel = kernel.cpu().numpy()
    return no_blur_image

def multiple_blur_kernel(input_image, kernel_size, sigma1, sigma2):
    mean1, mean2 = 3, 1
    variance1, variance2 = sigma1**2, sigma2**2

    channels = input_image.size() [0]

    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_cord = torch.arange(kernel_size)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    channels = input_image.size() [0]

    gaussian_kernel1 = (1./(2.*math.pi*variance1)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean1)**2., dim=-1) /\
                        (2*variance1)
                    )
    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel1 = gaussian_kernel1 / torch.sum(gaussian_kernel1)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel1 = gaussian_kernel1.view(1, kernel_size, kernel_size)
    gaussian_kernel1 = gaussian_kernel1.repeat(channels, 1, 1, 1)
    gaussian_kernel1 = gaussian_kernel1.cuda()

    gaussian_kernel2 = (1./(2.*math.pi*variance2)) *\
                    torch.exp(
                        -torch.sum((xy_grid - mean2)**2., dim=-1) /\
                        (2*variance2)
                    )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel2 = gaussian_kernel2 / torch.sum(gaussian_kernel2)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel2 = gaussian_kernel2.view(1, kernel_size, kernel_size)
    gaussian_kernel2 = gaussian_kernel2.repeat(channels, 1, 1, 1)
    gaussian_kernel2 = gaussian_kernel2.cuda()

    padding = gaussian_kernel1.shape[-1] - 1
    k3 = torch.conv2d(gaussian_kernel2, gaussian_kernel1,
                        padding=padding)[:1]
    k3 = k3.permute(1, 0, 2, 3)
    channels, size = k3.size() [0], k3.size()[-1]
    multiple_kernel = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=(kernel_size, kernel_size),
                                      groups=channels, bias=False, padding=0,
                                      stride=1, device=input_image.device)

    multiple_kernel.weight.data = k3
    multiple_kernel.weight.requires_grad = False

    output_image = multiple_kernel(input_image) * 255

    # Measurement noise added
    measurement_noise = torch.normal(0, 10, size=output_image.size())
    measurement_noise = measurement_noise.cuda()

    output_image = output_image + measurement_noise
    output_image = output_image.clamp(0, 255).to(torch.uint8)

    output_image = output_image / 255
    return output_image

def uniform_blur_filter(input_image, kernel_size):
    # Create a uniform blur filter
    kernel = torch.ones(1, 1, kernel_size, kernel_size)
    kernel = kernel / (kernel_size * kernel_size)

    channels = input_image.size() [0]

    # Reshape to 2d depthwise convolutional weight
    kernel = kernel.view(1, kernel_size, kernel_size)
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.cuda()

    uniform_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                     kernel_size=(kernel_size, kernel_size),
                                     groups=channels, bias=False, padding=0,
                                     stride=1, device=input_image.device)

    uniform_filter.weight.data = kernel

    uniform_filter.weight.requires_grad = False
    output_image = uniform_filter(input_image) * 255

    # Measurement noise added
    measurement_noise = torch.normal(0, 10, size=output_image.size())
    measurement_noise = measurement_noise.cuda()

    output_image = output_image + measurement_noise
    output_image = output_image.clamp(0, 255).to(torch.uint8)

    output_image = output_image / 255
    return output_image

def gaussian_blur_filter(input_image, kernel_size, sigma):
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

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=(kernel_size, kernel_size),
                                      groups=channels, bias=False, padding=0,
                                      stride=1, device=input_image.device)

    gaussian_filter.weight.data = gaussian_kernel

    gaussian_filter.weight.requires_grad = False
    output_image = gaussian_filter(input_image) * 255

    # Measurement noise added
    measurement_noise = torch.normal(0, 10, size=output_image.size())
    measurement_noise = measurement_noise.cuda()

    output_image = output_image + measurement_noise
    output_image = output_image.clamp(0, 255).to(torch.uint8)

    output_image = output_image / 255

    return output_image

def motion_blur_filter(input_image):
    # Use custom motion blur
    channels, kernel_size = input_image.size() [0], 5

    # 45 degrees, 135 degrees
    motion_blur_data = torch.tensor([[0.,0.,0.,0.00414365, 0.],
                                    [0.01104972, 0.16298343, 0.02348066, 0., 0.00414365],
                                    [0.00690608, 0.13259669, 0.19475138, 0.0980663,  0.],
                                    [0., 0., 0.03453039, 0.18370166, 0.09530387],
                                    [0., 0.00552486, 0.,0.01104972, 0.03176796]])
    motion_blur_data = motion_blur_data.cuda()
    motion_blur_data = motion_blur_data / torch.sum(motion_blur_data)

    motion_blur_data = motion_blur_data.view(1, kernel_size, kernel_size)
    motion_blur_data = motion_blur_data.repeat(channels, 1, 1, 1)
    motion_blur_data = motion_blur_data.cuda()

    # Save weights in the object
    motion_blur_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                         kernel_size=(kernel_size, kernel_size),
                                         groups=channels, bias=False, padding=0,
                                         stride=1, device=input_image.device)

    motion_blur_filter.weight.data = motion_blur_data

    motion_blur_filter.weight.requires_grad = False
    output_image = motion_blur_filter(input_image) * 255

    # Measurement noise added
    measurement_noise = torch.normal(0, 10, size=output_image.size())
    measurement_noise = measurement_noise.cuda()

    output_image = output_image + measurement_noise
    output_image = output_image.clamp(0, 255).to(torch.uint8)

    output_image = output_image / 255

    return output_image

def batch_process(batch_path, blur_function, positional_arguments):
    """
    Blur images in batches ie blur multiple images at once using the same blur kernel type
    """
    batch_images = np.load(batch_path) ['arr_0']
    blurred_images = []
    for i in range(batch_images.shape[0]):
        img_tensor_i = transform(batch_images[i]).cuda()

        # Save it
        img_save = img_tensor_i.permute(1, 2, 0)
        img_save = img_save.cpu().numpy() * 255
        img_save = img_save.astype(np.uint8)
        im = Image.fromarray(img_save)
        im.save(f"imgs/dataset_generated/gen_{i}.jpg")

        blurred_image = blur_function(img_tensor_i, *positional_arguments)


        blurred_images.append(blurred_image)

    noisy_tensor = torch.stack(blurred_images, dim=0)
    return noisy_tensor

# Create parser for different blur tasks
blur_parser = create_parser()
args = blur_parser.parse_args()

# Load image

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

if args.load_img_path is not None:
    img = Image.open(args.load_img_path)

    # Apply transformation
    img_tensor = transform(img).cuda()

# Kernel size
kernel_size = args.kernel_size

# Create a kernel based on the input argument
if args.gaussian:
    if args.batch:
        noisy_tensor = batch_process(args.batch_path, gaussian_blur_filter, [kernel_size, args.sigma])
    else:
        noisy_tensor = gaussian_blur_filter(img_tensor, kernel_size, args.sigma)
        noisy_tensor = noisy_tensor.unsqueeze(dim=0)
elif args.uniform:
    if args.batch:
        noisy_tensor = batch_process(args.batch_path, uniform_blur_filter, [kernel_size])
    else:
        noisy_tensor = uniform_blur_filter(img_tensor, kernel_size)
        noisy_tensor = noisy_tensor.unsqueeze(dim=0)
elif args.motion:
    noisy_tensor = motion_blur_filter(img_tensor)
    noisy_tensor = noisy_tensor.unsqueeze(dim=0)

elif args.multi:
    noisy_tensor = multiple_blur_kernel(img_tensor, args.kernel_size, args.sigma1, args.sigma2)
    noisy_tensor = noisy_tensor.unsqueeze(dim=0)

elif args.no_blur:
    noisy_tensor = no_blur_kernel(img_tensor, args.kernel_size)
    noisy_tensor = noisy_tensor.unsqueeze(dim=0)
else:
    NotImplementedError("Blur Kernel does not exist")


noisy_tensor = noisy_tensor.permute(0, 2, 3, 1)
noisy_tensor = noisy_tensor.cpu().numpy() * 255
noisy_tensor = noisy_tensor.astype(np.uint8)

# Save
for i in range(noisy_tensor.shape[0]):
    im = Image.fromarray(noisy_tensor[i])
    if args.save_img_path is None:
        img_save_path = f"imgs/dataset_generated_blur/blur_{i}.jpg"
    else:
        img_save_path = args.save_img_path
    im.save(img_save_path)

    img_copy = Image.open(img_save_path)
    img_copy_tensor = transform(img_copy).cuda()
