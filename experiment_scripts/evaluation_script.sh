#!/bin/bash

HOME_DIR="${HOME}/denoising_diffusion"
KERNEL_TYPE="multi_kernel"
for i in 1; do
    # Construct the input file name
    deblur_path="${HOME_DIR}/results/${KERNEL_TYPE}/example_${i}/example_${i}_deblur_adir_learn_loaded.jpg"
    blur_path="${HOME_DIR}/results/${KERNEL_TYPE}/example_${i}/example_${i}_blur.jpg"
    gt_path="${HOME_DIR}/results/${KERNEL_TYPE}/example_${i}/example_${i}_gt.jpg"

    # Send to function
    deblur_blur_args="--evaluation_metric --test_img_path ${deblur_path} --reference_img_path ${blur_path}"
    gt_blur_args="--evaluation_metric --test_img_path ${gt_path} --reference_img_path ${blur_path}"
    deblur_gt_args="--evaluation_metric --test_img_path ${deblur_path} --reference_img_path ${gt_path}"

    # Call the Python script with the input file
    echo "Deblur vs Blur"
    python3 $HOME_DIR/load_image.py $deblur_blur_args
    echo "GT vs Blur"
    python3 $HOME_DIR/load_image.py $gt_blur_args
    echo "Deblur vs GT"
    python3 $HOME_DIR/load_image.py $deblur_gt_args
done
