import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from PIL import Image
import os

# Example custom dataset for deepfake detection
default_label_map = {"real": 0, "fake": 1}

class DeepfakeDataset(Dataset):
    def __init__(self, image_dir, label_map=default_label_map, processor=None):
        self.image_dir = image_dir
        self.label_map = label_map
        self.processor = processor or AutoImageProcessor.from_pretrained('facebook/deit-small-patch16-224')
        self.samples = []
        for label in os.listdir(image_dir):
            label_path = os.path.join(image_dir, label)
            if os.path.isdir(label_path):
                for img_name in os.listdir(label_path):
                    self.samples.append((os.path.join(label_path, img_name), self.label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(label)
        return inputs

# Paths to your dataset
data_dir = "./deepfake_data"  # Should contain 'real/' and 'fake/' subfolders

# Load processor and model
processor = AutoImageProcessor.from_pretrained('facebook/deit-small-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'facebook/deit-small-patch16-224',
    num_labels=2,
    id2label={0: "real", 1: "fake"},
    label2id={"real": 0, "fake": 1}
)

dataset = DeepfakeDataset(data_dir, processor=processor)

# Split dataset (simple split, replace with your own logic as needed)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

def collate_fn(batch):
    return {
        k: torch.stack([item[k] for item in batch])
        for k in batch[0]
    }

training_args = TrainingArguments(
    output_dir="./deepfake_vit_finetuned",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    data_collator=collate_fn,
)

if __name__ == "__main__":
    trainer.train()
    trainer.save_model("./deepfake_vit_finetuned")
