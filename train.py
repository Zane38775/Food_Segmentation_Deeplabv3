import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from data_prep import FoodSegDataset
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import f1_score
import numpy as np
import torch.nn.functional as F

# 设置设备
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 定义超参数
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 100
NUM_WORKERS = 4
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False

# 设置数据路径
DATA_DIR = r'C:\Users\Zane\Documents\UROP24_FoodSeg\data\FoodSeg103'
IMG_DIR = os.path.join(DATA_DIR, 'Images', 'img_dir')
MASK_DIR = os.path.join(DATA_DIR, 'Images', 'ann_dir')

# 设置类别数量
NUM_CLASSES = 104  # 103 food classes + 1 background class

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

def get_model(num_classes):
    model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

def train_fn(loader, model, optimizer, loss_fn, scaler, epoch, writer):
    model.train()
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}")
    total_loss = 0
    start_time = time.time()
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.long().to(device=DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)['out']
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())
        total_loss += loss.item()

        # Log to TensorBoard
        writer.add_scalar('Loss/train', loss.item(), epoch * len(loader) + batch_idx)

    avg_loss = total_loss / len(loader)
    end_time = time.time()
    print(f"Training for epoch {epoch+1} completed in {end_time - start_time:.2f} seconds")
    return avg_loss

def check_accuracy(loader, model, device=DEVICE):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()
    all_preds = []
    all_targets = []

    start_time = time.time()
    print("Starting validation...")
    with torch.no_grad():
        for i, (x, y) in enumerate(tqdm(loader, desc="Validating")):
            x = x.to(device)
            y = y.to(device)
            preds = torch.argmax(model(x)['out'], dim=1)
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds == y).sum()) / (
                (preds == y).sum() + (y == y).sum() + 1e-8
            )
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())

            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{len(loader)} batches")

    print("Calculating final metrics...")
    accuracy = num_correct / num_pixels
    dice = dice_score / len(loader)
    
    print("Calculating F1 score...")
    try:
        f1 = f1_score(all_targets, all_preds, average='weighted')
    except Exception as e:
        print(f"Error calculating F1 score: {e}")
        print(f"all_targets shape: {np.shape(all_targets)}")
        print(f"all_preds shape: {np.shape(all_preds)}")
        print(f"all_targets unique values: {np.unique(all_targets)}")
        print(f"all_preds unique values: {np.unique(all_preds)}")
        f1 = 0.0  # 设置一个默认值

    end_time = time.time()
    print(f"Validation completed in {end_time - start_time:.2f} seconds")
    return accuracy.item(), dice.item(), f1

def main():
    print(f"Using device: {DEVICE}")

    train_transform = A.Compose(
        [
            A.RandomResizedCrop(height=IMAGE_HEIGHT, width=IMAGE_WIDTH, scale=(0.8, 1.0)),
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.Transpose(),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=None),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.5),
            A.OneOf([
                A.HueSaturationValue(),
                A.RandomBrightnessContrast(),
                A.ColorJitter(),
            ], p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ],
        is_check_shapes=False
    )

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ],
        is_check_shapes=False
    )

    model = get_model(num_classes=NUM_CLASSES).to(DEVICE)
    loss_fn = FocalLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.1)

    train_dataset = FoodSegDataset(DATA_DIR, IMG_DIR, MASK_DIR, split='train', transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    val_dataset = FoodSegDataset(DATA_DIR, IMG_DIR, MASK_DIR, split='test', transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

    if LOAD_MODEL:
        print("Loading checkpoint...")
        checkpoint = torch.load("my_checkpoint.pth.tar")
        model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])

    scaler = torch.cuda.amp.GradScaler()
    writer = SummaryWriter()

    best_f1 = 0
    patience = 15
    patience_counter = 0

    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch: {epoch+1}")
        epoch_start_time = time.time()
        train_loss = train_fn(train_loader, model, optimizer, loss_fn, scaler, epoch, writer)

        # check accuracy
        print("Starting accuracy check...")
        try:
            accuracy, dice, f1 = check_accuracy(val_loader, model, DEVICE)
            print(f"Accuracy: {accuracy:.4f}, Dice: {dice:.4f}, F1: {f1:.4f}")

            # Log to TensorBoard
            writer.add_scalar('Loss/train_epoch', train_loss, epoch)
            writer.add_scalar('Accuracy/val', accuracy, epoch)
            writer.add_scalar('Dice/val', dice, epoch)
            writer.add_scalar('F1/val', f1, epoch)

            # Learning rate scheduler step
            scheduler.step(f1)
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Current learning rate: {current_lr}")

            # Early stopping logic
            if f1 > best_f1:
                best_f1 = f1
                patience_counter = 0
                # save best model
                checkpoint = {
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                }
                torch.save(checkpoint, "best_model.pth.tar")
                print(f"New best model saved with F1: {f1:.4f}")
            else:
                patience_counter += 1
                print(f"F1 did not improve. Patience: {patience_counter}/{patience}")

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        except Exception as e:
            print(f"Error during validation: {e}")
            print("Skipping this epoch and continuing...")

        epoch_end_time = time.time()
        print(f"Epoch {epoch+1} completed in {epoch_end_time - epoch_start_time:.2f} seconds")

    print("Training completed.")

    # 保存最终模型
    final_checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(final_checkpoint, "final_model.pth.tar")
    print("Final model saved as 'final_model.pth.tar'")

    writer.close()

if __name__ == "__main__":
    main()