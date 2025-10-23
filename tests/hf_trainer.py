import json
import torch
import evaluate
from torch import nn
from torchvision.transforms import ColorJitter
from transformers import (
    AutoFeatureExtractor,
    SegformerForSemanticSegmentation,
    TrainingArguments,
    Trainer,
)
from huggingface_hub import hf_hub_download

# =======================
# 1. 全局参数配置
# =======================
MODEL_CHECKPOINT = "nvidia/mit-b0"
DATASET_REPO = "segments/sidewalk-semantic"
LABEL_JSON = "id2label.json"
EPOCHS = 50
LEARNING_RATE = 6e-5
BATCH_SIZE = 2
HUB_MODEL_ID = "segformer-b0-finetuned-segments-sidewalk-2"

# =======================
# 2. 读取标签映射
# =======================
id2label_path = hf_hub_download(DATASET_REPO, LABEL_JSON, repo_type="dataset")
with open(id2label_path, "r") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}
label2id = {v: k for k, v in id2label.items()}
num_labels = len(id2label)

# =======================
# 3. 加载特征提取器和评估指标
# =======================
feature_extractor = AutoFeatureExtractor.from_pretrained(MODEL_CHECKPOINT)
metric = evaluate.load("mean_iou")
jitter = ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.1)

# =======================
# 4. 数据集加载与增强
# =======================
# 假设你已准备好`ds`，它是一个datasets对象
# 例如：ds = load_dataset("segments/sidewalk-semantic")
# 这里只是展示transform和split部分
ds = ds.shuffle(seed=1)
ds = ds["train"].train_test_split(test_size=0.2)
train_ds = ds["train"]
val_ds = ds["test"]

def train_transforms(example_batch):
    images = [jitter(x) for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    return feature_extractor(images, labels)

def val_transforms(example_batch):
    images = [x for x in example_batch['pixel_values']]
    labels = [x for x in example_batch['label']]
    return feature_extractor(images, labels)

train_ds.set_transform(train_transforms)
val_ds.set_transform(val_transforms)

# =======================
# 5. 评估函数
# =======================
def compute_metrics(eval_pred):
    with torch.no_grad():
        logits, labels = eval_pred
        logits_tensor = torch.from_numpy(logits)
        logits_tensor = nn.functional.interpolate(
            logits_tensor,
            size=labels.shape[-2:],
            mode="bilinear",
            align_corners=False,
        ).argmax(dim=1)
        pred_labels = logits_tensor.detach().cpu().numpy()
        metrics = metric._compute(
            predictions=pred_labels,
            references=labels,
            num_labels=num_labels,
            ignore_index=0,
            reduce_labels=feature_extractor.reduce_labels,
        )
        # 展开每类别指标
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()
        metrics.update({f"accuracy_{id2label[i]}": v for i, v in enumerate(per_category_accuracy)})
        metrics.update({f"iou_{id2label[i]}": v for i, v in enumerate(per_category_iou)})
        return metrics

# =======================
# 6. 模型和训练参数
# =======================
model = SegformerForSemanticSegmentation.from_pretrained(
    MODEL_CHECKPOINT,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
    use_safetensors=True
)

training_args = TrainingArguments(
    output_dir="segformer-b0-finetuned-outputs", 
    learning_rate=LEARNING_RATE,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    save_total_limit=3,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_steps=20,
    eval_steps=20,
    logging_steps=1,
    eval_accumulation_steps=5,
    load_best_model_at_end=True,
    push_to_hub=True,
    hub_model_id=HUB_MODEL_ID,
    hub_strategy="end",
)

# =======================
# 7. Trainer构建与训练
# =======================
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=feature_extractor,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

if __name__ == "__main__":
    trainer.train()
