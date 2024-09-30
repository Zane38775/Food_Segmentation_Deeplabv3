# Food Segmentation with DeepLabV3

This project implements a food segmentation model using the DeepLabV3 architecture with a ResNet101 backbone. It's designed to accurately segment food items in images, utilizing the FoodSeg103 dataset.

## Project Structure
![屏幕截图 2024-09-30 105122](https://github.com/user-attachments/assets/efff61d1-6d21-49ce-a434-1cd0a81839f6)


## Model Architecture

The project uses a modified DeepLabV3 architecture with ResNet101 backbone for food segmentation, implemented in `food_seg.py`.

### FoodSeg Model Features:

- Based on DeepLabV3 with ResNet101 backbone
- Pretrained on ImageNet
- Customized for food segmentation task with 104 classes (FoodSeg103 dataset)

### Model Structure:

1. **Base Model:**
   - Uses `deeplabv3_resnet101` pretrained on ImageNet

2. **Customization:**
   - Replaces the final classifier layer to match the number of food classes (104)

3. **Forward Pass:**
   - Utilizes the modified DeepLabV3 model for inference

### Key Components:

```python
def get_model(num_classes):
    model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model
```

## Dataset

The `FoodSegDataset` class in `data_prep.py` handles data loading and preprocessing for the FoodSeg103 dataset.

### Dataset Features:
- Supports train/val splits defined in text files
- Loads images and corresponding masks
- Applies customizable data augmentations

### Data Augmentation:
- Training data augmentations include:
  - Resizing
  - Rotation (up to 35 degrees)
  - Horizontal flip (50% chance)
  - Vertical flip (10% chance)
  - Normalization
- Validation data only undergoes resizing and normalization

## Inference

The main segmentation process is implemented in `main_seg.py`.

### Features:
- Loads a trained model from 'best_model.pth.tar'
- Processes images from the 'result/input/' folder
- Generates and saves segmentation results in 'result/output/'
- Supports various image formats (PNG, JPG, JPEG)
- Resizes input images to 256x256
- Uses a random color map for visualizing different food classes

## Usage

### Dataset Preparation:
```python
from dataset import FoodSegDataset, get_train_transform, get_val_transform

data_dir = 'path/to/FoodSeg103'
img_dir = 'path/to/FoodSeg103/Images/img_dir'
mask_dir = 'path/to/FoodSeg103/Images/ann_dir'

train_transform = get_train_transform(256, 256)
train_dataset = FoodSegDataset(data_dir, img_dir, mask_dir, split='train', transform=train_transform)

val_transform = get_val_transform(256, 256)
val_dataset = FoodSegDataset(data_dir, img_dir, mask_dir, split='val', transform=val_transform)
```

### Inference:
1. Place your input images in the result/input/ folder.
2. Run the segmentation script:
```python
python src/main_seg.py
```
3. The segmented images will be saved in the result/output/ folder.

## Requirements

1. Python
2. PyTorch
3. torchvision
4. OpenCV (cv2)
5. Pillow
6. numpy
7. albumentations
   
Install the required packages using:
```python
pip install torch torchvision opencv-python Pillow numpy albumentations
```

## Running the Project

1. Prepare your dataset in the `data/FoodSeg103/` directory.
   Please download the file from url and unzip the data in ./data folder (./data/FoodSeg103/), with passwd: LARCdataset9947
   https://research.larc.smu.edu.sg/downloads/datarepo/FoodSeg103.zip
3. Download a trained model saved as ''best_model.pth.tar'' in the project root.
   https://drive.google.com/file/d/17_X_Fx2yECX1ob9ECnzND98nokACorve/view?usp=sharing
4. Place input images for segmentation in `result/input/`.
5. Run `main_seg.py` to perform segmentation.
6. Check the segmented images in `result/output/`.'

