import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from PIL import Image
import os

# Update this path to where you extract the Kaggle dataset
DATASET_DIR = './ff-c23'

# The dataset is expected to have 'real' and 'fake' subfolders with images
LABEL_MAP = {'real': 0, 'fake': 1}

class FFC23Dataset(Dataset):
    def __init__(self, root_dir, processor=None, label_map=LABEL_MAP):
        self.root_dir = root_dir
        self.processor = processor or AutoImageProcessor.from_pretrained('facebook/deit-small-patch16-224')
        self.samples = []
        for label in label_map:
            label_dir = os.path.join(root_dir, label)
            if os.path.isdir(label_dir):
                for fname in os.listdir(label_dir):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(label_dir, fname), label_map[label]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        inputs = self.processor(images=image, return_tensors='pt')
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs['labels'] = torch.tensor(label)
        return inputs

# Load processor and model
processor = AutoImageProcessor.from_pretrained('facebook/deit-small-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'facebook/deit-small-patch16-224',
    num_labels=2,
    id2label={0: 'real', 1: 'fake'},
    label2id={'real': 0, 'fake': 1}
)

dataset = FFC23Dataset(DATASET_DIR, processor=processor)

# Simple train/val split
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

def collate_fn(batch):
    return {k: torch.stack([item[k] for item in batch]) for k in batch[0]}

training_args = TrainingArguments(
    output_dir='./ff_c23_vit_finetuned',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    logging_dir='./logs_ff_c23',
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model='eval_loss',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=processor,
    data_collator=collate_fn,
)

if __name__ == '__main__':
    trainer.train()
    trainer.save_model('./ff_c23_vit_finetuned')
