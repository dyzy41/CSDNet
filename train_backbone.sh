#!/bin/bash

# 指定只使用第0号和第2号 GPU
export CUDA_VISIBLE_DEVICES=4,5

# python main.py --dataset SYSU-CD --model_type cd --model_arch SEED --model_name CSDNet --backbone resnet50 --exp_name resnet50 --max_steps 40000 --batch_size 16 --devices 2 --strategy auto --accelerator gpu --src_size 256 --lr 0.0003

# python main.py --dataset SYSU-CD --model_type cd --model_arch SEED --model_name CSDNet --backbone efficientnet_b5 --exp_name efficientnet_b5 --max_steps 40000 --batch_size 16 --devices 2 --strategy auto --accelerator gpu --src_size 256 --lr 0.0003

# python main.py --dataset SYSU-CD --model_type cd --model_arch SEED --model_name CSDNet --backbone hrnet_w18 --exp_name hrnet_w18 --max_steps 40000 --batch_size 16 --devices 2 --strategy auto --accelerator gpu --src_size 256 --lr 0.0003

# python main.py --dataset SYSU-CD --model_type cd --model_arch SEED --model_name CSDNet --backbone hrnet_w32 --exp_name hrnet_w32 --max_steps 40000 --batch_size 16 --devices 2 --strategy auto --accelerator gpu --src_size 256 --lr 0.0003

python main.py --dataset SYSU-CD --model_type cd --model_arch SEED --model_name CSDNet --backbone convnext_base --exp_name convnext_base --max_steps 30000 --batch_size 16 --devices 2 --strategy ddp --accelerator gpu --src_size 256 --lr 0.0003

python main.py --dataset SYSU-CD --model_type cd --model_arch SEED --model_name CSDNet --backbone mambaout_base --exp_name mambaout_base --max_steps 30000 --batch_size 16 --devices 2 --strategy ddp --accelerator gpu --src_size 256 --lr 0.0003

# python main.py --dataset WHUCD --model_type cd --model_arch SEED --model_name CSDNet --backbone swinv2_base_window8_256 --exp_name swinv2_base_window8_256 --max_steps 40000 --batch_size 16 --devices 2 --strategy auto --accelerator gpu --src_size 256 --lr 0.0003 --work_dirs work_dirs_WHUCD

# python main.py --dataset WHUCD --model_type cd --model_arch SEED --model_name CSDNet --backbone swinv2_small_window8_256 --exp_name swinv2_small_window8_256 --max_steps 40000 --batch_size 16 --devices 2 --strategy auto --accelerator gpu --src_size 256 --lr 0.0003 --work_dirs work_dirs_WHUCD