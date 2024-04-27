import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
import torchvision.transforms as t
from PIL import Image
import cv2
import random
import torch.nn.functional as F


img = cv2.imread('/content/sunset-1373171_1280.jpg')
image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(image)

data_transforms = t.Compose([
    t.Resize((224, 224)),
    t.ToTensor(),
    t.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

def viewCropped(m,Adash):
  alpha = []
  regularizedFeatmap = 0

  gap_deno = torch.zeros(1,1)
  for i in range(1,m):
    gap_deno += torch.mean(Adash[:,i-1:i,:,:], dim = (2,3))

  for i in range(1,m):
    gap_nume = torch.mean(Adash[:, i-1:i, :, :], dim=(2,3))
    alpha.append((gap_nume/gap_deno))

  for i in range(1,m-1):
    regularizedFeatmap += alpha[i] * Adash[:,i-1:i,:,:]
  return regularizedFeatmap


"""
flood_fill() func used from an online source
"""
def flood_fill(binary_mask, start_pixel):
    connectivity = torch.zeros_like(binary_mask)
    region_size = 0

    stack = [start_pixel]

    while stack:
        pixel = stack.pop()
        x, y = pixel

        if (x < 0 or x >= binary_mask.shape[2] or
            y < 0 or y >= binary_mask.shape[3] or
            binary_mask[0, 0, x, y] == 0 or
            connectivity[0, 0, x, y] == 1):
            continue

        connectivity[0, 0, x, y] = 1
        region_size += 1

        stack.append((x + 1, y))
        stack.append((x - 1, y))
        stack.append((x, y + 1))
        stack.append((x, y - 1))

    return connectivity, region_size

class FG(nn.Module):
  def __init__(self):
    super().__init__()

    self.model = models.resnet50(pretrained=True)
    self.model = nn.Sequential(*list(self.model.children())[:-2])

    self.conv1 = nn.Conv2d(2048,2048,1)
    self.bn1 = nn.BatchNorm2d(2048)
    self.relu1 = nn.ReLU()

  def forward(self, img):
    F = self.model(img)
    A = self.relu1(self.bn1(self.conv1(F)))
    F1 = F * A
    pool = torch.mean(F1, dim=(2, 3))
    Fr = torch.cat((pool,pool),dim = 1).reshape(-1,1)
    m = random.randrange(1,2048)
    Adash = A[:,:m, : , :]
    regularizedFeatmap = viewCropped(m,Adash)
    return regularizedFeatmap,A,Fr

fg = FG()
img_transform = data_transforms(img).unsqueeze(dim=0)
regularizedFeatmap,A,Fr = fg(img_transform)

Mc = F.interpolate(regularizedFeatmap, size=(224,224), mode='bilinear', align_corners=False)

Mc_flatten = Mc.view(-1)
sorted_vals = np.sort(Mc_flatten.detach().numpy())
percentile_threshold = np.percentile(sorted_vals, q=95)
threshold = percentile_threshold * torch.max(Mc)
Mc_filter = torch.where(Mc < threshold, torch.tensor(0.0), (1.0))

start_pixel = (0, 0)
connectivity, region_size = flood_fill(Mc_filter, start_pixel)
I_crop = img_transform * connectivity
_,_,Fc = fg(I_crop)



eraseRandom = random.randrange(2,2048)
Adash2 = A[:, eraseRandom-1:eraseRandom, :, :]
Me = F.interpolate(Adash2, size=(224,224), mode='bilinear', align_corners=False)
Me_flatten = Me.view(-1)
sorted_vals = np.sort(Me_flatten.detach().numpy())
percentile_threshold = np.percentile(sorted_vals, q=95)

threshold = percentile_threshold * torch.max(Me)
Me_filter = torch.where(Me < threshold, torch.tensor(1.0), (0.0))
I_erase = img_transform * Me_filter
_,_,Fe = fg(I_erase)