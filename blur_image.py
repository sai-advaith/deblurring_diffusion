import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# def uniform_blur_filter(input_image):
#     kernel_size = 5
#     kernel = torch.ones(1, 1, kernel_size, kernel_size)
#     kernel = kernel / (kernel_size * kernel_size)

#     channels = input_image.size() [1]
#     kernel = kernel.expand((3, channels, kernel_size, kernel_size))
#     kernel = kernel.cuda()

#     blurred_image = F.conv2d(input_image, kernel, padding=kernel_size//2)
#     blurred_image = blurred_image.clamp(0, 1)
#     measurement_noise = torch.normal(0, 10, size=blurred_image.size())
#     measurement_noise = measurement_noise.cuda()

#     blurred_image = blurred_image + measurement_noise
#     return blurred_image

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

import math
def uniform_blur_filter(input_image):
    kernel_size = 5
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

def apply_blur(input_image):
    # Set these to whatever you want for your gaussian filter
    kernel_size = 5
    sigma = 0.5

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
                                      kernel_size=kernel_size, groups=channels, bias=False,
                                      padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel

    gaussian_filter.weight.requires_grad = False
    output_image = gaussian_filter(input_image)
    return output_image

# Load image
img = Image.open("7_256.jpg")

# Define transformation
transform = transforms.Compose([
    transforms.ToTensor(),
])

# Apply transformation
img_tensor = transform(img).cuda()


# Check pixel value range
print(torch.min(img_tensor), torch.max(img_tensor))

noisy_tensor = uniform_blur_filter(img_tensor)
print(noisy_tensor.size())
noisy_tensor = noisy_tensor.permute(1, 2, 0)
noisy_tensor = noisy_tensor.cpu().numpy() * 255
noisy_tensor = noisy_tensor.astype(np.uint8)

print(noisy_tensor.shape)

im = Image.fromarray(noisy_tensor)
im.save("7_256_blur.jpg")

img_copy = Image.open("7_256_blur.jpg")
img_copy_tensor = transform(img_copy).cuda()
print(img_copy_tensor.size())