#!/bin/bash

args1_deblur_blur="--evaluation_metric --test_img_path imgs/guided_diffusion_256_gen_deblur2_10.jpg --reference_img_path imgs/guided_diffusion_256_gen_blur2.jpg"
args2_deblur_gt="--evaluation_metric --test_img_path imgs/guided_diffusion_256_gen_deblur2_10.jpg --reference_img_path imgs/guided_diffusion_256_gen.jpg"
args3_blur_gt="--evaluation_metric --test_img_path imgs/guided_diffusion_256_gen_blur2.jpg --reference_img_path imgs/guided_diffusion_256_gen.jpg"

python3 load_image.py $args1_deblur_blur
python3 load_image.py $args2_deblur_gt
python3 load_image.py $args3_blur_gt
