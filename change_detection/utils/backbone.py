import timm
import os
import torch



def build_backbone(model_name='resnet'):
    dim_change = False
    if model_name=='resnet50':
        try:
            model = timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'resnet50.a1_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model(model_name, pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [256, 512, 1024, 2048], 'out_channels': 256, 'num_outs': 4}
        num_stages = 5

    elif model_name=='efficientnet_b5':
        try:
            model = timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'efficientnet_b5.sw_in12k_ft_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model(model_name, pretrained=False, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [40, 64, 176, 512], 'out_channels': 256, 'num_outs': 4}
        num_stages = 5

    elif model_name=='hrnet_w18':
        try:
            model = timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'hrnet_w18.ms_aug_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model(model_name, pretrained=False, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 1024], 'out_channels': 256, 'num_outs': 4}
        num_stages = 5

    elif model_name=='hrnet_w32':
        try:
            model = timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'hrnet_w32.ms_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model(model_name, pretrained=False, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 1024], 'out_channels': 256, 'num_outs': 4}
        num_stages = 5

    elif model_name=='convnext_base':
        try:
            model = timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'convnext_base.clip_laion2b_augreg_ft_in12k_in1/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model(model_name, pretrained=False, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 1024], 'out_channels': 256, 'num_outs': 4}
        num_stages = 4

    elif model_name=='hrnet_w64':
        try:
            model = timm.create_model('hrnet_w64', pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'hrnet_w64.ms_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model(model_name, pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 1024], 'out_channels': 256, 'num_outs': 4}
        num_stages = 5 

    elif model_name=='mambaout_base':
        try:
            model = timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'mambaout_base.in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model(model_name, pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 768], 'out_channels': 256, 'num_outs': 4}
        num_stages = 4
        dim_change = True

    elif model_name=='swinv2_base_window8_256':
        try:
            model = timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'swinv2_base_window8_256.ms_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model(model_name, pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [128, 256, 512, 1024], 'out_channels': 256, 'num_outs': 4}
        num_stages = 4
        dim_change = True

    elif model_name=='swinv2_small_window8_256':
        try:
            model = timm.create_model(model_name, pretrained=True, pretrained_cfg_overlay=dict(file=os.path.join(os.environ.get("PRETRAIN"), 'swinv2_small_window8_256.ms_in1k/pytorch_model.bin')), features_only=True)
        except:
            model = timm.create_model(model_name, pretrained=True, features_only=True)
        FPN_DICT = {'type': 'FPN', 'in_channels': [96, 192, 384, 768], 'out_channels': 256, 'num_outs': 4}
        num_stages = 4
        dim_change = True

    return model, num_stages, FPN_DICT, dim_change
    

    