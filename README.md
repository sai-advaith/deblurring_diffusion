# Blind Inverse Problem with Diffusion models

## Blur Image

Use an input image and blur it using the code in `blur_image.py`. The blurring code can be run with the following argument

`python3 blur_image.py --kernel_size 3 --sigma1 10 --sigma2 10 --multi --load_img_path LOAD_IMG_PATH --save_img_path SAVE_IMG_PATH`

Can choose between the following blur kernels
1. Gaussian
2. Uniform
3. Multiple
4. Motion Blur
5. No Blur

