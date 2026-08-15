import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(111)         

DATASET = "data"

IMG_SIZE_REAL = 128
IMG_SIZE_PIXEL = 128
BATCH_SIZE = 4
EPOCHS = 200
LR = 0.0002

# Dataset

class Pix2PixDataset(Dataset):
    def __init__(self, root):
        self.real_dir = os.path.join(root, "real")
        self.pixel_dir = os.path.join(root, "pixel")

        self.files = sorted([
            f for f in os.listdir(self.real_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        self.transform_real = transforms.Compose([
            transforms.Resize((IMG_SIZE_REAL, IMG_SIZE_REAL), interpolation=InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.transform_pixel_rgb = transforms.Compose([
            transforms.Resize((IMG_SIZE_PIXEL, IMG_SIZE_PIXEL), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3)
        ])

        self.transform_alpha = transforms.Compose([
            transforms.Resize((IMG_SIZE_PIXEL, IMG_SIZE_PIXEL), interpolation=InterpolationMode.NEAREST),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        real = Image.open(os.path.join(self.real_dir, name)).convert("RGB")
        real = self.transform_real(real)

        pixel = Image.open(os.path.join(self.pixel_dir, name)).convert("RGBA")
        rgb = self.transform_pixel_rgb(pixel.convert("RGB"))
        alpha = self.transform_alpha(pixel.getchannel("A"))

        pixel = torch.cat([rgb, alpha], dim=0)

        return real, pixel


# Generator

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc0 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(nn.Conv2d(32, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True))
        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True))
        self.down4 = nn.Sequential(nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))
        self.down5 = nn.Sequential(nn.Conv2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True))

        self.up1 = nn.Sequential(nn.ConvTranspose2d(512, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.up2 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.Dropout(0.5))
        self.up3 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.up4 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.up5 = nn.Sequential(nn.ConvTranspose2d(128, 32, 4, 2, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))

        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.final_rgb = nn.Conv2d(64, 3, 3, 1, 1)
        self.final_alpha = nn.Conv2d(64, 1, 3, 1, 1)

    def forward(self, x):
        e0 = self.enc0(x)

        d1 = self.down1(e0)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        u1 = self.up1(d5)
        u1 = torch.cat([u1, d4], dim=1)

        u2 = self.up2(u1)
        u2 = torch.cat([u2, d3], dim=1)

        u3 = self.up3(u2)
        u3 = torch.cat([u3, d2], dim=1)

        u4 = self.up4(u3)
        u4 = torch.cat([u4, d1], dim=1)

        u5 = self.up5(u4)
        u5 = torch.cat([u5, e0], dim=1)

        feat = self.refine(u5)

        rgb = torch.tanh(self.final_rgb(feat))
        alpha = torch.sigmoid(self.final_alpha(feat) * 5)

        return torch.cat([rgb, alpha], dim=1)


# Discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(7, 64, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, real, pixel):
        x = torch.cat([real, pixel], dim=1)
        return self.model(x)


# Training pt

train_dataset = Pix2PixDataset(os.path.join(DATASET, "train"))
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

opt_G = torch.optim.Adam(G.parameters(), lr=LR, betas=(0.5, 0.999))
opt_D = torch.optim.Adam(D.parameters(), lr=LR, betas=(0.5, 0.999))

bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

for epoch in range(EPOCHS):
    print(f"epoch: {epoch}")
    loop = tqdm(train_loader)

    for real, pixel in loop:
        real = real.to(DEVICE)
        pixel = pixel.to(DEVICE)

        fake = G(real)

        # Train D
        D_real = D(real, pixel)
        D_fake = D(real, fake.detach())

        loss_D = (bce(D_real, torch.ones_like(D_real)) + 
                  bce(D_fake, torch.zeros_like(D_fake))) / 2

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train G
        D_fake = D(real, fake)

        loss_G_adv = bce(D_fake, torch.ones_like(D_fake))
        loss_G_l1 = 100 * l1(fake[:, :3, :, :], pixel[:, :3, :, :])

        alpha_fake = fake[:, 3:4, :, :]
        alpha_real = pixel[:, 3:4, :, :]

        loss_alpha_l1 = 20 * l1(alpha_fake, alpha_real)
        loss_alpha_bin = torch.mean((alpha_fake * (1 - alpha_fake)) ** 2)

        loss_alpha = loss_alpha_l1 + 10 * loss_alpha_bin
        loss_G = loss_G_adv + loss_G_l1 + loss_alpha

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loop.set_postfix(
            loss_D=loss_D.item(),
            loss_G=loss_G.item()
        )

torch.save(G.state_dict(), "generator_u5_128.pth")
print("Training finished")