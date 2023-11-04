#!/bin/bash

# Set pythonpath
HOME="/home/sanketh/denoising_diffusion"
KERNEL_TYPE="multi_kernel"
export PYTHONPATH=$PYTHONPATH:$HOME

diffusion_steps=1500
for i in 1; do
    file_name="example_${i}"
    # GT and blur image
    dir_path="${HOME}/results/${KERNEL_TYPE}/${file_name}"
    gt_img_path="${dir_path}/${file_name}_gt.jpg"
    blur_img_path="${dir_path}/${file_name}_blur.jpg"

    # Sampling arguments
    sample_args="--attention_resolutions 32,16,8 --class_cond False --diffusion_steps ${diffusion_steps} --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --clip_denoised True --classifier_scale 15.0 --classifier_path ${HOME}/models/256x256_classifier.pt --classifier_depth 2 --model_path ${HOME}/models/256x256_diffusion_uncond.pt --batch_size 1 --num_samples 1 --timestep_respacing ${diffusion_steps} --kernel_size 5 --image_path ${blur_img_path} --batch_load 1 --idx_low 82 --idx_high 89 --data_dir ~/denoising_diffusion/imgs/dataset_generated_blur --wandb_log False --use_ddim False --gpu_device_num ${i-1}"

    nohup python3 $HOME/scripts/classifier_sample.py $sample_args > $dir_path/output_$i.txt&
done