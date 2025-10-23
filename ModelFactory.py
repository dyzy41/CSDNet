import torch
import os
import cv2
import numpy as np
from torchmetrics.classification import Accuracy, ConfusionMatrix
import lightning as L
import torch.nn as nn
from utils.metric import CM2Metric, save_metrics
import torch.optim as optim
from change_detection import build_model
from torch.optim.lr_scheduler import *
import csv
import torch.nn.functional as F
from change_detection.utils.domain_genelization_loss import *
from change_detection.utils.loss_func import coral_loss


# 1. 定义 Lightning 模型
class BaseCD(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        if args.resize_size > 1:
            self.example_input_array = torch.randn((2, 6, args.resize_size, args.resize_size))
        else:
            self.example_input_array = torch.randn((2, 6, args.crop_size, args.crop_size))
        # define parameters
        self.save_hyperparameters(args)

        self.hyparams = args
        if self.hyparams.resize_size > 1:
            self.if_slide = False
        else:
            self.if_slide = self.hyparams.src_size > self.hyparams.crop_size
        self.save_test_results = os.path.join(self.hyparams.work_dirs, self.hyparams.exp_name+'_TrainingFiles', self.hyparams.save_test_results)

        # model training
        self.change_detection = build_model(self.hyparams.model_name)
        self.val_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.hyparams.num_classes)
        self.test_confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=self.hyparams.num_classes)
        if self.hyparams.loss_type == 'ce':
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.BCEWithLogitsLoss()
        self.val_loss_epoch = []
        
        # prepare test output directory once
        self.test_output_dir = os.path.join(self.hyparams.work_dirs, f"{self.hyparams.exp_name}_TrainingFiles",
                                            self.hyparams.save_test_results)
        os.makedirs(self.test_output_dir, exist_ok=True)

    def forward(self, x):
        xA, xB = x[:, :3], x[:, 3:]
        out = self.change_detection(xA, xB)
        if (isinstance(out, tuple) or isinstance(out, list)) is False:
            out = [out]
        return out

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        x, y = batch['imgAB'], batch['lab']
        outs = self(x)
        loss = self._loss(outs, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, pathA, pathB = batch['imgAB'], batch['lab'], batch['pathA'], batch['pathB']
        logits, val_loss_step = self._infer(x, y)
        self.val_loss_epoch.append(val_loss_step)
        self.val_confusion_matrix.update(self._logits2preds(logits), y)

    def on_validation_epoch_end(self):
        # 在所有 batch 更新完成后，compute 出整 epoch 的混淆矩阵
        cm = self.val_confusion_matrix.compute().cpu().numpy()
        metrics = CM2Metric(cm)
        val_loss_epoch = torch.mean(torch.stack(self.val_loss_epoch))
        self.log('val_loss', val_loss_epoch, prog_bar=True, on_epoch=True, logger=True, sync_dist=True)
        self.log_dict({
            'val_oa': metrics[0],
            'val_iou': metrics[4][1],
            'val_f1': metrics[5][1],
            'val_recall': metrics[6][1],
            'val_precision': metrics[7][1]
        }, prog_bar=True, on_epoch=True, on_step=False, logger=True, sync_dist=True)
        # 重置，为下一个 epoch 准备
        self.val_confusion_matrix.reset()
        self.val_loss_epoch = []

    def test_step(self, batch: dict, batch_idx: int) -> None:
        x, y, pathA, pathB = batch['imgAB'], batch['lab'], batch['pathA'], batch['pathB']
        logits, test_loss = self._infer(x, y)
        self.test_confusion_matrix.update(self._logits2preds(logits), y)

        pred_np = self._logits2preds(logits).cpu().numpy().astype('uint8')
        for p, mask in zip(pathA, pred_np):
            base = os.path.splitext(os.path.basename(p))[0] + '.png'
            out_path = os.path.join(self.test_output_dir, base)
            cv2.imwrite(out_path, (mask * 255).astype('uint8'))

    def on_test_epoch_end(self):
        cm = self.test_confusion_matrix.compute().cpu().numpy()
        metrics = CM2Metric(cm)
        self.log_dict({
            'test_oa': metrics[0],
            'test_iou': metrics[4][1],
            'test_f1': metrics[5][1],
            'test_recall': metrics[6][1],
            'test_precision': metrics[7][1]
        }, prog_bar=True, sync_dist=True)
        # 重置，为下一个 epoch 准备

        save_metrics(save_path=os.path.join(self.hyparams.work_dirs, self.hyparams.exp_name+'_TrainingFiles', os.path.basename(self.hyparams.exp_name)+'_metrics.csv'), metrics=metrics)

        self.test_confusion_matrix.reset()

    def _logits2preds(self, logits):
        """Convert logits to predictions."""
        if self.hyparams.loss_type == 'ce':
            preds = logits.argmax(dim=1)
        else:
            preds = torch.sigmoid(logits).round()
            preds = preds.squeeze(1)  # Remove the channel dimension for binary segmentation
        return preds

    def _loss(self, outs, y, state='train'):
        if state == 'train':
            if self.hyparams.loss_type == 'ce':
                loss = sum(w * self.criterion(o, y.long()) for w, o in zip(self.hyparams.loss_weights, outs))
            else:
                loss = sum(w * self.criterion(o, y.unsqueeze(1).float()) for w, o in zip(self.hyparams.loss_weights, outs))
        else:
            if self.hyparams.loss_type == 'ce':
                loss = self.criterion(outs, y.long())
            else:
                loss = self.criterion(outs, y.unsqueeze(1).float())
        return loss

    def _infer(self, x, y):
        """Run either sliding-window or single-pass inference."""

        if self.if_slide:
            logits, val_loss = self._slide_inference(x, y)
            return logits, val_loss
        else:
            outs = self(x)
            val_loss =  self._loss(outs, y)
            logits = outs[self.hyparams.pred_idx]
            return logits, val_loss

    def _slide_inference(self, inputs, labels):
        h_crop = w_crop = self.hyparams.crop_size
        h_stride = w_stride = getattr(self.hyparams, "overlap", h_crop // 2)
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.hyparams.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]
                outs = self(crop_img)
                crop_seg_logit = outs[self.hyparams.pred_idx]
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        val_loss = self._loss(seg_logits, labels, state='val')

        return seg_logits, val_loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hyparams.lr, weight_decay=1e-4)
        
        def lr_lambda(step: int) -> float:
            warmup = self.hyparams.warmup
            power = 3.0
            if step < warmup:
                raw = float(step) / float(max(1, warmup))
            else:
                progress = float(step - warmup) / float(max(1, self.hyparams.max_steps - warmup))
                raw = max(0.0, (1.0 - progress) ** power)
            min_factor = self.hyparams.min_lr / self.hyparams.lr
            return max(raw, min_factor)
        
        scheduler = LambdaLR(optimizer, lr_lambda) 

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                # "monitor": "val_iou",
                "strict": False,
                "frequency": 1,
                "name": None
            },
        }


class SEED(BaseCD):
    """
    SEED Lightning 模型：
    - 继承自 BaseCD，复用数据加载、训练/验证/测试流程
    - 只需要在 __init__ 中替换 change_detection 网络即可
    """
    def __init__(self, args):
        # 调用父类，完成超参保存、confusion matrix、criterion 等初始化
        super().__init__(args)

    def _loss(self, outs, y, state='train'):
        """Compute the loss for the current batch."""
        if state == 'train':
            loss = sum(w * self.criterion(o, y.long())
                        for w, o in zip(self.hyparams.loss_weights, outs))
            loss = loss/2.0
        else:
            loss = self.criterion(outs, y.long())
        return loss

    def _infer(self, x, y):
        """Run either sliding-window or single-pass inference."""

        if self.if_slide and self.hyparams.model_type != 'dgcd':
            logits, val_loss = self._slide_inference(x, y)
            return logits, val_loss
        else:
            outs = self(x)
            val_loss =  self._loss(outs, y)
            logits = (outs[0]+outs[1])/2.0
            return logits, val_loss

    def _slide_inference(self, inputs, labels):

        h_crop = w_crop = self.hyparams.crop_size
        h_stride = w_stride = getattr(self.hyparams, "overlap", h_crop // 2)
        batch_size, _, h_img, w_img = inputs.size()
        out_channels = self.hyparams.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = inputs.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = inputs.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = inputs[:, :, y1:y2, x1:x2]

                outs = self(crop_img)
                crop_seg_logit = (outs[0]+outs[1])/2.0

                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        seg_logits = preds / count_mat

        val_loss = self._loss(seg_logits, labels, state='val')

        return seg_logits, val_loss

