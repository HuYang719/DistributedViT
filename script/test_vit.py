import json
from PIL import Image
import torch
from torchvision import transforms
from torchsummary import summary
from time import time
import numpy as np

torch.set_printoptions(profile="full")

# Load ViT



from pytorch_pretrained_vit import ViT

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')
print('Using device:', device)

model = ViT('B_16_imagenet1k', pretrained=True)
model.eval()
# print(model)
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
# print(torch.get_num_threads(),torch.get_num_interop_threads())


# Load image
# NOTE: Assumes an image `img.jpg` exists in the current directory
img = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)), 
    # transforms.Normalize(0.5, 0.5),
])(Image.open('/home/lucyyang/Documents/04-pytorh_ViT/PyTorch-Pretrained-ViT/img.jpg')).unsqueeze(0)
print("image shape is ", img.shape) # torch.Size([1, 3, 384, 384])
# print("Input image value is:\n", img)


start = time()
with torch.no_grad():
    outputs = model(img).squeeze(0)
print("direct output shape is", outputs.shape)  # (1, 1000)
print(time() - start)

# Load class names
labels_map = json.load(open('/home/lucyyang/Documents/04-pytorh_ViT/PyTorch-Pretrained-ViT/examples/simple/labels_map.txt'))
labels_map = [labels_map[str(i)] for i in range(1000)]

print('-----')
for idx in torch.topk(outputs, k=3).indices.tolist():
    prob = torch.softmax(outputs, -1)[idx].item()
    res = torch.softmax(outputs, -1)
    # print(res)
    # print(res[388])
    print('[{idx}] {label:<75} ({p:.2f}%)'.format(idx=idx, label=labels_map[idx], p=prob*100))