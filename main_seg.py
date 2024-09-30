import torch
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import models
import torch.nn as nn
import os

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
NUM_CLASSES = 104

# Define the model
def get_model(num_classes):
    model = models.segmentation.deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
    model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
    return model

# Load the trained model
model = get_model(NUM_CLASSES).to(DEVICE)
checkpoint = torch.load('best_model.pth.tar', map_location=DEVICE)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

# Define the image transformation
transform = A.Compose(
    [
        A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ],
)

# Set the input and output folder paths
input_folder = r'C:\Users\Zane\Documents\UROP24_FoodSeg\result\input'
output_folder = r'C:\Users\Zane\Documents\UROP24_FoodSeg\result\output'

# Get the list of image files in the input folder
image_files = [file for file in os.listdir(input_folder) if file.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Process each image file
for image_file in image_files:
    # Load and preprocess the image
    image_path = os.path.join(input_folder, image_file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transformed_image = transform(image=image)['image']
    input_image = transformed_image.unsqueeze(0).to(DEVICE)

    # Perform inference
    with torch.no_grad():
        output = model(input_image)['out']
        prediction = torch.argmax(output, dim=1)
        prediction = prediction.squeeze().cpu().numpy()

    # Convert the prediction to RGB color map
    color_map = np.random.randint(0, 255, size=(NUM_CLASSES, 3), dtype=np.uint8)
    rgb_prediction = color_map[prediction]

    # Save the segmentation result
    output_path = os.path.join(output_folder, f'segmented_{image_file}')
    cv2.imwrite(output_path, cv2.cvtColor(rgb_prediction, cv2.COLOR_RGB2BGR))
    print(f"Segmentation result for '{image_file}' saved as '{output_path}'")