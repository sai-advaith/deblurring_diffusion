"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os
from PIL import Image
import math

import numpy as np
import matplotlib.pyplot as plt
import torch as th
import torch.distributed as dist
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
def get_no_blur_filter(kernel_size):
    with th.no_grad():
        kernel = th.zeros((kernel_size, kernel_size), dtype=th.float32)
        kernel[kernel_size // 2, kernel_size // 2] = 1.0

        # Reshape to 2d depthwise convolutional weight
        kernel = kernel.view(1, kernel_size, kernel_size)
        kernel = kernel.repeat(3, 1, 1, 1)
        kernel = kernel.cuda()

        return kernel

def load_kernel_path(kernel_path):
    with th.no_grad():
        kernel = th.load(kernel_path)
        kernel = kernel.cuda()
        return kernel

def get_multi_filter(kernel_size, sigma1, sigma2):
    with th.no_grad():
        mean1, mean2 = 3.0, 1.0
        gaussian_kernel1 = get_gaussian_filter(kernel_size - 2, sigma1, mean1)
        gaussian_kernel2 = get_gaussian_filter(kernel_size - 2, sigma2, mean2)

        padding = gaussian_kernel2.shape[-1] - 1
        multi_kernel = th.conv2d(gaussian_kernel2, gaussian_kernel1, padding=padding) [:1]
        multi_kernel = multi_kernel.permute(1, 0, 2, 3)
        return multi_kernel

def init_blur_kernel(kernel_size, timesteps, std):
    """
    Initialize with a smooth / uniform operator
    """
    kernel_uniform = get_uniform_filter(kernel_size)
    kernel = kernel_uniform + th.randn_like(kernel_uniform) * std
    print(kernel)

    cosine_similarity = th.nn.CosineSimilarity(dim=0, eps=1e-6)
    cosine_kernels = cosine_similarity(kernel, kernel_uniform)

    plt.imshow(cosine_kernels.cpu().numpy()[0], cmap='viridis', interpolation='nearest')
    plt.colorbar()
    plt.savefig(f'/home/sanketh/denoising_diffusion/imgs/cosine_similarity_kernels/cosine_similarity_kernels_{timesteps}.jpg')
    plt.close()

    return kernel

def get_uniform_filter(kernel_size):
    with th.no_grad():
        kernel = th.ones(1, kernel_size, kernel_size)
        kernel = kernel / (kernel_size * kernel_size)

        # Reshape to 2d depthwise convolutional weight
        kernel = kernel.repeat(3, 1, 1, 1)
        kernel = kernel.cuda()

        return kernel

def get_motion_filter(kernel_size):
    with th.no_grad():
        # 45 degrees, 135 degrees
        motion_blur_data = th.tensor([[0.,0.,0.,0.00414365, 0.],
                                        [0.01104972, 0.16298343, 0.02348066, 0., 0.00414365],
                                        [0.00690608, 0.13259669, 0.19475138, 0.0980663,  0.],
                                        [0., 0., 0.03453039, 0.18370166, 0.09530387],
                                        [0., 0.00552486, 0.,0.01104972, 0.03176796]])

        motion_blur_data = motion_blur_data.cuda()
        motion_blur_data = motion_blur_data / th.sum(motion_blur_data)

        motion_blur_data = motion_blur_data.view(1, kernel_size, kernel_size)
        motion_blur_data = motion_blur_data.repeat(3, 1, 1, 1)
        motion_blur_data = motion_blur_data.cuda()

        return motion_blur_data

def get_saved_kernel(kernel_path):
    blur_kernel = th.load(kernel_path)
    return blur_kernel.cuda()

def get_gaussian_filter(kernel_size, sigma, mean=-1):
    """
    Given kernel size, output the blur filter
    """
    with th.no_grad():
        # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
        x_cord = th.arange(kernel_size)
        x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
        y_grid = x_grid.t()
        xy_grid = th.stack([x_grid, y_grid], dim=-1)

        if mean == -1:
            mean = (kernel_size - 1)/2.
        variance = sigma**2.

        # Calculate the 2-dimensional gaussian kernel which is
        # the product of two gaussian distributions for two different
        # variables (in this case called x and y)
        gaussian_kernel = (1./(2.*math.pi*variance)) *\
                        th.exp(
                            -th.sum((xy_grid - mean)**2., dim=-1) /\
                            (2*variance)
                        )
        # Make sure sum of values in gaussian kernel equals 1.
        gaussian_kernel = gaussian_kernel / th.sum(gaussian_kernel)

        # Reshape to 2d depthwise convolutional weight
        gaussian_kernel = gaussian_kernel.view(1, kernel_size, kernel_size)

        # 3 = Num channels
        gaussian_kernel = gaussian_kernel.repeat(3, 1, 1, 1)
        gaussian_kernel = gaussian_kernel.cuda()

        return gaussian_kernel

def get_corrupted_image(image_path):
    """
    Given image path, load the data into a tensor 
    """
    # Define transformation
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    # Load image
    img = Image.open(image_path)
    img_tensor = transform(img).to(dist_util.dev())
    img_tensor = (img_tensor * 2) - 1

    return img_tensor.unsqueeze(dim=0)

def get_corrupted_batch(data_dir, idx_low, idx_high, image_path=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Load images as batch
    images = []
    for i in range(idx_low, idx_high+1):
        image_path_i = f"{data_dir}/blur_{i}.jpg"
        img_i = Image.open(image_path_i)
        img_tensor_i = transform(img_i).to(dist_util.dev())
        img_tensor_i = (img_tensor_i * 2) - 1
        images.append(img_tensor_i)

    # # TODO: Remove this, only for experimentation purposes
    # if image_path is not None:
    #     images.append(get_corrupted_image(image_path).squeeze(dim=0))

    return th.stack(images, dim=0)

def main():
    args = create_argparser().parse_args()

    th.cuda.set_device(args.gpu_device_num)
    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")

    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    # TODO: Adjust according to Table 12
    classifier.load_state_dict(
        dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()
    logger.log("classifer loaded")

    assert args.batch_load == args.batch_size == args.num_samples

    # Creating uniform blur kernel of size kernel size x kernel size
    if args.batch_load == 1:
        corrupted_image = get_corrupted_image(args.image_path)
    else:
        corrupted_image = get_corrupted_batch(args.data_dir, args.idx_low,
                                              args.idx_high, args.image_path)


    # TODO: Make 0.05 an argument
    std = 0.05
    # blur_kernel = init_blur_kernel(args.kernel_size, args.diffusion_steps, std)
    blur_kernel = load_kernel_path('/home/sanketh/denoising_diffusion/results/multi_kernel/example_1/final_predict_kernel.pt')
    # blur_kernel = get_gaussian_filter(args.kernel_size, sigma=15)
    # blur_kernel = get_multi_filter(args.kernel_size, 10, 10)
    # blur_kernel = get_uniform_filter(args.kernel_size)

    def cond_fn(x, t, y=None):
        assert y is not None
        with th.enable_grad():
            x_in = x.detach().requires_grad_(True)
            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        if args.use_ddim:
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                noise=corrupted_image,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                device=dist_util.dev(),
            )
        else:
            sample = sample_fn(
                model_fn,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
                cond_fn=cond_fn,
                adaptive_diffusion=True,
                kernel=blur_kernel,
                noise=None,
                corrupted_image=corrupted_image,
                device=dist_util.dev(),
                gradient_scaling=args.classifier_scale,
                wandb_log=args.wandb_log
            )
        # Back to [0, 255] range
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        all_labels.extend([labels.cpu().numpy() for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    if dist.get_rank() == 0:
        shape_str = "x".join([str(x) for x in arr.shape])
        out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
        logger.log(f"saving to {out_path}")
        np.savez(out_path, arr, label_arr)

    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        kernel_size=5,
        image_path=None,
        batch_load=-1,
        idx_low=-1,
        idx_high=-1,
        data_dir=None,
        wandb_log=False,
        gpu_device_num=0
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
