CUDA_VISIBLE_DEVICES=4 python main.py --dataset WHUCD --model_type cd --model_arch SEED --model_name CSDNet --exp_name CSDNet_HRNet_WHUCD --max_steps 40000 --batch_size 16 --devices 1 --strategy auto --accelerator gpu --src_size 256 --lr 0.0003 --work_dirs work_dirs_best --mode test_loader

CUDA_VISIBLE_DEVICES=4 python main.py --dataset MSRSCD --model_type cd --model_arch SEED --model_name CSDNet --exp_name CSDNet_HRNet_MSRSCD --max_steps 40000 --batch_size 16 --devices 1 --strategy auto --accelerator gpu --src_size 1024 --crop_size 256 --lr 0.0003 --work_dirs work_dirs_best --mode test_loader

CUDA_VISIBLE_DEVICES=4 python main.py --dataset LEVIR-CD --model_type cd --model_arch SEED --model_name CSDNet --exp_name CSDNet_HRNet_LEVIR --max_steps 40000 --batch_size 16 --devices 1 --strategy auto --accelerator gpu --src_size 1024 --crop_size 256 --lr 0.0003 --work_dirs work_dirs_best --mode test_loader

CUDA_VISIBLE_DEVICES=4 python main.py --dataset SYSU-CD --model_type cd --model_arch SEED --model_name CSDNet --exp_name CSDNet_HRNet_SYSU --max_steps 40000 --batch_size 16 --devices 1 --strategy auto --accelerator gpu --src_size 256 --lr 0.0003 --work_dirs work_dirs_best --mode test_loader

CUDA_VISIBLE_DEVICES=4 python main.py --dataset S2Looking --model_type cd --model_arch SEED --model_name CSDNet --exp_name CSDNet_HRNet_S2Looking --max_steps 40000 --batch_size 16 --devices 1 --strategy auto --accelerator gpu --src_size 1024 --crop_size 256 --lr 0.0003 --work_dirs work_dirs_best --mode test_loader