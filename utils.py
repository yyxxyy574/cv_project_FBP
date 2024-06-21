from torchvision import transforms
from PIL import Image
import numpy as np

def num_flat_features(x):
    sizes = x.size()[1: ]
    num_flat_features = 1
    for size in sizes:
        num_flat_features *= size
    
    return num_flat_features

# 加载图片，进行缩放、归一化等预处理
def load_image(path):
    # 缩放到适当尺寸、转化为tensor并归一化，便于数据读取与训练
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[1, 1, 1])
    ])
    # 将图像数据转化为Tensor
    img = Image.open(path)
    input_tensor = transform(img)
    # 将图像数据转化为numpy数组
    img = transforms.Resize((224, 224))(img)
    input_array = np.array(img).astype(np.float32) / 255.
    return input_tensor, input_array