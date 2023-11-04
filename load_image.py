import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import piqa
import lpips
import warnings

import torch
import torchvision
import torchvision.transforms as T
import torch.nn.functional as F


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
        im.save(f'{image_save_path}')

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

def transform_img(img_path, permute=True, expand_range=False):
    # Define transformation
    transform = T.Compose([
        T.ToTensor(),
    ])

    img = Image.open(img_path)
    img_tensor = transform(img)
    if expand_range:
        img_tensor = img_tensor.cuda() * 255.0
        img_tensor = img_tensor.to(torch.uint8)
    if permute:
        img_tensor = img_tensor.permute(1, 2, 0)

    return img_tensor

def resize(img_tensor, reshape_size):
    img_max, img_min = img_tensor.max(), img_tensor.min()
    resized_tensor = F.interpolate(img_tensor, reshape_size, mode='bicubic')
    return resized_tensor.clamp(img_min, img_max)

def measure_eval(image_path1, image_path2):
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch")
    img1_tensor = transform_img(image_path1, permute=False).unsqueeze(dim=0).cpu()
    img2_tensor = transform_img(image_path2, permute=False).unsqueeze(dim=0).cpu()

    if img1_tensor.size() != img2_tensor.size():
        # Resize the smaller image
        if img1_tensor.numel() > img2_tensor.numel():
            img2_tensor = resize(img2_tensor, img1_tensor.size()[-2:])
        else:
            img1_tensor = resize(img1_tensor, img2_tensor.size()[-2:])

    assert img1_tensor.size() == img2_tensor.size()

    # LPIPS
    lpips_fn = piqa.LPIPS()
    lpips_val = lpips_fn(img1_tensor, img2_tensor).item()

    # PSNR, SSIM
    psnr_fn, ssim_fn = piqa.PSNR(), piqa.SSIM()
    psnr_val = psnr_fn(img1_tensor, img2_tensor)
    ssim_val = ssim_fn(img1_tensor, img2_tensor)

    # print(lpips_val.size())
    print(f"PSNR between {image_path1} and {image_path2} is: {psnr_val}")
    print(f"SSIM between {image_path1} and {image_path2} is: {ssim_val}")
    print(f"LPIPS between {image_path1} and {image_path2} is: {lpips_val}\n")

    # Restore warnings (if needed)
    warnings.resetwarnings()
    return psnr_val, ssim_val, lpips_val

# Create parser for different utility tasks
util_parser = load_parser()

args = util_parser.parse_args()

# rescale('imagenet.JPEG')
if args.load_image:
    load(args.load_image_path, args.load_image_save_path)

if args.evaluation_metric:
    measure_eval(args.test_img_path, args.reference_img_path)

# plot('copy.txt')