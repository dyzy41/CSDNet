import os
import sys
import argparse
import torch
import types

# 确保项目根目录在 sys.path，方便 import ModelFactory 等模块
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# 在导入 ModelFactory 之前尽量劫持 timm.create_model，强制不下载预训练权重（避免 HF 下载阻塞）
try:
    import importlib
    timm_spec = importlib.util.find_spec("timm")
    if timm_spec is not None:
        import timm
        _orig_timm_create = getattr(timm, 'create_model', None)
        if _orig_timm_create is not None:
            def _create_model_no_pretrained(name, pretrained=False, *a, **k):
                return _orig_timm_create(name, pretrained=False, *a, **k)
            timm.create_model = _create_model_no_pretrained
except Exception:
    # 忽略劫持失败
    pass

# 尝试导入 thop
try:
    from thop import profile, clever_format
except Exception as e:
    raise ImportError("请先安装 thop：python -m pip install thop") from e

# 导入项目中的 ModelFactory（要求 ROOT 已加入 sys.path）
import ModelFactory


def build_args_for_instantiate(cli):
    """
    构造 argparse.Namespace 供 ModelFactory/Lightning 使用。
    若 cli.src_size 为 None，则使用 cli.crop_size 作为 src_size。
    """
    src_size_val = cli.src_size if getattr(cli, 'src_size', None) is not None else getattr(cli, 'crop_size', 256)

    defaults = {
        'dataset': getattr(cli, 'dataset', 'LEVIR-CD'),
        'model_name': getattr(cli, 'model_name', 'CSDNet'),
        'model_type': getattr(cli, 'model_type', 'cd'),
        'resize_size': getattr(cli, 'resize_size', 1),
        'src_size': src_size_val,
        'crop_size': getattr(cli, 'crop_size', 256),
        'work_dirs': getattr(cli, 'work_dirs', 'work_dirs'),
        'exp_name': getattr(cli, 'exp_name', 'debug'),
        'save_test_results': getattr(cli, 'save_test_results', 'test_results'),
        'num_classes': getattr(cli, 'num_classes', 2),
        'loss_type': getattr(cli, 'loss_type', 'ce'),
        'loss_weights': getattr(cli, 'loss_weights', [1.0, 1.0]),
        'pred_idx': getattr(cli, 'pred_idx', 0),
        'lr': getattr(cli, 'lr', 1e-4),
        'min_lr': getattr(cli, 'min_lr', 1e-5),
        'warmup': getattr(cli, 'warmup', 0),
        'max_steps': getattr(cli, 'max_steps', 1000),
        'model_arch': getattr(cli, 'model_arch', 'SEED'),
        'resume_path': getattr(cli, 'resume_path', None)
    }
    return argparse.Namespace(**defaults)


def try_profile(model, device, H, W, mode):
    """
    使用 thop.profile 统计 MACs 和 params。
    支持两种输入格式：pair 或 concat（fallback）。
    返回 (macs, params, used_mode)
    """
    model.eval()
    model.to(device)

    # 尝试 pair 输入
    if mode == 'pair':
        inA = torch.randn(1, 3, H, W, device=device)
        inB = torch.randn(1, 3, H, W, device=device)
        try:
            macs, params = profile(model, inputs=(inA, inB), verbose=False)
            return macs, params, 'pair'
        except Exception:
            # 尝试 concat
            pass

    # concat fallback
    in_concat = torch.randn(1, 6, H, W, device=device)
    try:
        macs, params = profile(model, inputs=(in_concat,), verbose=False)
        return macs, params, 'concat'
    except Exception as e:
        raise RuntimeError(f"thop 统计失败：尝试 pair/concat 均失败，错误：{e}")


def main():
    parser = argparse.ArgumentParser(description="Compute params and FLOPs for a model")
    parser.add_argument('--model_arch', type=str, default='SEED', help='Class name in ModelFactory.py (e.g. SEED)')
    parser.add_argument('--model_name', type=str, default='CSDNet', help='model_name passed to constructor')
    parser.add_argument('--crop_size', type=int, default=256, help='输入用于计算的 H/W（通常是 crop_size）')
    parser.add_argument('--src_size', type=int, default=None, help='源图像大小（若需要）')
    parser.add_argument('--device', type=str, default='cpu', help='cpu 或 cuda')
    parser.add_argument('--input_mode', type=str, choices=['concat', 'pair', 'auto'], default='auto',
                        help='输入格式：concat=1x6xHxW, pair=(1x3xHxW,1x3xHxW), auto 尝试 pair 再 concat')
    parser.add_argument('--cuda_device', type=str, default=None, help='可选：设置 CUDA_VISIBLE_DEVICES（例如 6）')
    parser.add_argument('--num_classes', type=int, default=2)
    args_cli = parser.parse_args()

    if args_cli.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args_cli.cuda_device)

    device = args_cli.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("检测不到 CUDA，改为 cpu")
        device = 'cpu'

    inst_args = build_args_for_instantiate(args_cli)

    ModelClass = getattr(ModelFactory, args_cli.model_arch, None)
    if ModelClass is None:
        raise AttributeError(f"ModelFactory 中找不到类 {args_cli.model_arch}")

    # 尝试实例化模型；若遇到 HF 下载相关错误再进行重试（再次劫持 timm）
    try:
        model = ModelClass(inst_args)
    except Exception as e:
        err_str = str(e)
        if any(k in err_str.lower() for k in ('huggingface', 'hf_hub', 'localentrynotfounderror', 'hf_hub_download', 'load_state_dict_from_hf')):
            print("检测到尝试从 HuggingFace 下载预训练权重失败，尝试再次禁用 timm pretrained 并重试 …")
            try:
                import importlib
                timm_spec = importlib.util.find_spec("timm")
                if timm_spec is not None:
                    import timm
                    orig_create = getattr(timm, 'create_model', None)
                    if orig_create is not None:
                        def create_model_no_pretrained(name, pretrained=False, *a, **k):
                            return orig_create(name, pretrained=False, *a, **k)
                        timm.create_model = create_model_no_pretrained
                model = ModelClass(inst_args)
            except Exception as e2:
                # 抛出二次错误以便用户查看
                raise e2 from e
        else:
            raise

    H = args_cli.crop_size
    W = args_cli.crop_size

    mode = args_cli.input_mode
    mode_try = 'pair' if mode == 'auto' else mode

    macs, params, used_mode = try_profile(model, device, H, W, mode_try)

    macs_fmt, params_fmt = clever_format([macs, params], "%.3f")
    flops = macs * 2
    flops_fmt = clever_format([flops], "%.3f")[0]

    print("Model class:", args_cli.model_arch)
    print("model_name:", args_cli.model_name)
    print("Input mode used:", used_mode)
    print(f"Input tensor: 1 x {'6' if used_mode == 'concat' else '3+3'} x {H} x {W}")
    print(f"Params: {params} ({params_fmt})")
    print(f"MACs: {macs} ({macs_fmt})")
    print(f"Estimated FLOPs (≈ 2*MACs): {flops} ({flops_fmt})")


if __name__ == '__main__':
    main()