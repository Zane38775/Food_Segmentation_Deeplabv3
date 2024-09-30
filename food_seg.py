import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

class FoodSeg(nn.Module):
    def __init__(self, num_classes):
        super(FoodSeg, self).__init__()
        # 加载预训练的 DeepLabV3 模型
        self.deeplab = deeplabv3_resnet50(weights='COCO_WITH_VOC_LABELS_V1')
        
        # 替换最后的分类器以匹配我们的类别数
        self.deeplab.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))
        
        # 初始化新的分类器层
        nn.init.xavier_uniform_(self.deeplab.classifier[4].weight)
        nn.init.zeros_(self.deeplab.classifier[4].bias)

    def forward(self, x):
        return self.deeplab(x)

def get_model(num_classes):
    return FoodSeg(num_classes)