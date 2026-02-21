
# Updated: Extract frames from videos and use them for training
import torch
from torch.utils.data import Dataset
from transformers import AutoImageProcessor, ViTForImageClassification, TrainingArguments, Trainer
from PIL import Image
import os
import cv2
import glob

DATASET_DIR = r"E:\dsu_projects\deepfake-forensic\FaceForensics++_C23"
REAL_DIR = os.path.join(DATASET_DIR, "original")
FAKE_DIRS = [
    os.path.join(DATASET_DIR, "Deepfakes"),
    os.path.join(DATASET_DIR, "FaceSwap"),
    os.path.join(DATASET_DIR, "Face2Face"),
    os.path.join(DATASET_DIR, "NeuralTextures"),
    os.path.join(DATASET_DIR, "DeepFakeDetection")
]

# Directory to store extracted frames
FRAME_DIR = os.path.join(DATASET_DIR, "frames")
os.makedirs(FRAME_DIR, exist_ok=True)

def extract_frames(video_path, out_dir, num_frames=5):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count == 0:
        cap.release()
        return []
    step = max(1, frame_count // num_frames)
    frames = []
    for i in range(0, frame_count, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame_file = os.path.join(out_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_file, frame)
            frames.append(frame_file)
        if len(frames) >= num_frames:
            break
    cap.release()
    return frames

def prepare_frame_dataset():
    samples = []
    # Real videos
    for vid in glob.glob(os.path.join(REAL_DIR, "*.mp4")):
        vid_name = os.path.splitext(os.path.basename(vid))[0]
        out_dir = os.path.join(FRAME_DIR, f"real_{vid_name}")
        frames = extract_frames(vid, out_dir)
        for f in frames:
            samples.append((f, 0))
    # Fake videos
    for fake_dir in FAKE_DIRS:
        for vid in glob.glob(os.path.join(fake_dir, "*.mp4")):
            vid_name = os.path.splitext(os.path.basename(vid))[0]
            out_dir = os.path.join(FRAME_DIR, f"fake_{vid_name}")
            frames = extract_frames(vid, out_dir)
            for f in frames:
                samples.append((f, 1))
    return samples

class DeepfakeFrameDataset(Dataset):
    def __init__(self, samples, processor=None):
        self.samples = samples
        self.processor = processor or AutoImageProcessor.from_pretrained('facebook/deit-small-patch16-224')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(label)
        return inputs

processor = AutoImageProcessor.from_pretrained('facebook/deit-small-patch16-224')
model = ViTForImageClassification.from_pretrained(
    'facebook/deit-small-patch16-224',
    num_labels=2,
    id2label={0: "real", 1: "fake"},
    label2id={"real": 0, "fake": 1},
    ignore_mismatched_sizes=True
)

samples = prepare_frame_dataset()
dataset = DeepfakeFrameDataset(samples, processor=processor)

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
