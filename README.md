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

## Denoising Code
You can denoise any input image (size 256 x 256) with the following command:

`python3 scripts/classifier_sample.py --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 100 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 True --use_scale_shift_norm True --clip_denoised True --classifier_scale 15.0 --classifier_path CLASSIFIER_PATH --classifier_depth 2 --model_path MODEL_PATH --batch_size 1 --num_samples 1 --timestep_respacing 100 --kernel_size 5 --image_path IMAGE_PATH --batch_load 1 --idx_low 82 --idx_high 89 --data_dir DATA_DIRECTORY --wandb_log False --use_ddim False --gpu_device_num 0 --init_std 0.05 --wandb_username WANDB_USERNAME --wandb_project_name WANDB_PROJECT_NAME --wandb_run_name WANDB_RUN_NAME`

Note: Tweak parameters to modify deblurring strength.

## References

This code is based on based on [guided-diffusion](https://github.com/openai/guided-diffusion). Setup conda environment based on that code.