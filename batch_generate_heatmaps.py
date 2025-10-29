import os
import subprocess
import argparse

def read_test_samples(txt_path, num_samples=5):
    """读取测试集样本路径"""
    with open(txt_path, 'r') as f:
        lines = f.readlines()[:num_samples]
    
    samples = []
    for line in lines:
        parts = line.strip().split('  ')
        if len(parts) >= 2:
            img_a, img_b = parts[0], parts[1]
            samples.append((img_a, img_b))
    
    return samples

def main(args):
    datasets = {
        'WHUCD': {
            'checkpoint': 'work_dirs_best/CSDNet_HRNet_WHUCD_TrainingFiles/best-model-step=step=038000-iou=val_iou=0.9083.ckpt',
            'img_size': 256
        },
        'SYSU-CD': {
            'checkpoint': 'work_dirs_best/CSDNet_HRNet_SYSU_TrainingFiles/best-model-step=step=012000-iou=val_iou=0.6739.ckpt',
            'img_size': 256
        }
    }
    
    for dataset_name, config in datasets.items():
        print(f"\n{'='*50}")
        print(f"Processing {dataset_name} dataset")
        print(f"{'='*50}\n")
        
        # 构建数据集路径
        dataset_root = os.path.join(os.environ.get("CDPATH", ""), dataset_name)
        test_txt = os.path.join(dataset_root, 'test.txt')
        
        if not os.path.exists(test_txt):
            print(f"Warning: {test_txt} not found, skipping...")
            continue
        
        # 读取测试样本
        samples = read_test_samples(test_txt, num_samples=args.num_samples)
        print(f"Found {len(samples)} samples to process")
        
        # 为每个样本生成热力图
        for idx, (img_a, img_b) in enumerate(samples):
            print(f"\nProcessing sample {idx+1}/{len(samples)}")
            print(f"  Time A: {img_a}")
            print(f"  Time B: {img_b}")
            
            # 构建命令
            cmd = [
                'python', 'infer.py',
                '--checkpoint_path', config['checkpoint'],
                '--img_a', img_a,
                '--img_b', img_b,
                '--save_dir', f"heatmap_results/{dataset_name}",
                '--img_size', str(config['img_size']),
                '--alpha', str(args.alpha)
            ]
            
            # 执行命令
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error processing sample {idx+1}: {e}")
                continue
    
    print("\n" + "="*50)
    print("All heatmaps generated successfully!")
    print("="*50)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch generate heatmaps for multiple datasets')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to process per dataset')
    parser.add_argument('--alpha', type=float, default=0.5,
                       help='Transparency of heatmap overlay (0-1)')
    
    args = parser.parse_args()
    main(args)