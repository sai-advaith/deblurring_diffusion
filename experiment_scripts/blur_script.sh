#!/bin/bash

filter_args="no_blur_kernel"

HOME_DIR="/${HOME}/advaith/denoising_diffusion"
blur_args="--no_blur --kernel_size 5"
for i in 1 2 3; do
    file_name="example_${i}"
    load_img_path="${HOME_DIR}/results/${filter_args}/${file_name}/${file_name}_gt.jpg"
    save_img_path="${HOME_DIR}/results/${filter_args}/${file_name}/${file_name}_blur.jpg"
    script_args="${blur_args} --save_img_path ${save_img_path} --load_img_path ${load_img_path}"
    python3 $HOME_DIR/blur_image.py $script_args
done