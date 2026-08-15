import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET = "data"

IMG_SIZE_REAL = 128
IMG_SIZE_PIXEL = 32
BATCH_SIZE = 4
EPOCHS = 200
LR = 0.0002

# Dataset

class Pix2PixDataset(Dataset):

    def __init__(self, root):

        self.real_dir = os.path.join(root, "real")
        self.pixel_dir = os.path.join(root, "pixel")

        self.files = [f for f in os.listdir(self.real_dir)
                      if f.lower().endswith((".png",".jpg",".jpeg"))]

        self.transform_real = transforms.Compose([
            transforms.Resize((IMG_SIZE_REAL, IMG_SIZE_REAL)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.transform_pixel = transforms.Compose([
            transforms.Resize((IMG_SIZE_PIXEL, IMG_SIZE_PIXEL)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

        self.transform_alpha = transforms.Compose([
            transforms.Resize((IMG_SIZE_PIXEL, IMG_SIZE_PIXEL)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        name = self.files[idx]

        real = Image.open(os.path.join(self.real_dir, name)).convert("RGB")
        real = self.transform_real(real)

        pixel = Image.open(os.path.join(self.pixel_dir, name)).convert("RGBA")

        rgb = self.transform_pixel(pixel.convert("RGB"))
        alpha = self.transform_alpha(pixel.getchannel("A"))

        pixel = torch.cat([rgb, alpha], dim=0)

        return real, pixel


class WeakCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32,64,3,1,1),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU()
        )

        self.out = nn.Conv2d(64,4,3,1,1)

    def forward(self,x):

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = self.out(x)

        rgb = torch.tanh(x[:,:3,:,:])
        alpha = torch.sigmoid(x[:,3:4,:,:])  # ❌ 無 sharpen

        return torch.cat([rgb, alpha], dim=1)

# Training pt

train_dataset = Pix2PixDataset(os.path.join(DATASET,"train"))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = WeakCNN().to(DEVICE)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
l1 = nn.L1Loss()

for epoch in range(EPOCHS):

    print("epoch:", epoch)
    loop = tqdm(train_loader)

    for real, pixel in loop:

        real = real.to(DEVICE)
        pixel = pixel.to(DEVICE)

        fake = model(real)

        fake = torch.nn.functional.interpolate(fake, size=(32,32), mode='bilinear')

        loss = l1(fake, pixel)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

torch.save(model.state_dict(), "weak_cnn.pth")
print("🔥CNN Training finished")