import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET = "dataset_ModelA"

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 300
LR = 1e-4


# --------------------
# Dataset
# --------------------

class Pix2PixDataset(Dataset):

    def __init__(self, root):

        self.real_dir = os.path.join(root, "real")
        self.pixel_dir = os.path.join(root, "pixel")

        self.files = os.listdir(self.real_dir)

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        name = self.files[idx]

        real = Image.open(os.path.join(self.real_dir, name)).convert("RGB")
        pixel = Image.open(os.path.join(self.pixel_dir, name)).convert("RGB")

        real = self.transform(real)
        pixel = self.transform(pixel)

        return real, pixel


# --------------------
# Generator (U-Net)
# --------------------

class Generator(nn.Module):

    def __init__(self):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(3,64,4,2,1),
            nn.LeakyReLU(0.2)
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )

        self.down3 = nn.Sequential(
            nn.Conv2d(128,256,4,2,1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128,64,4,2,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64,3,4,2,1),
            nn.Tanh()
        )

    def forward(self,x):

        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)

        u1 = self.up1(d3)
        u2 = self.up2(u1)
        out = self.up3(u2)

        return out


# --------------------
# Discriminator
# --------------------

class Discriminator(nn.Module):

    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(

            nn.Conv2d(6,64,4,2,1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64,128,4,2,1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128,1,4,1,1)
        )

    def forward(self,real,pixel):

        x = torch.cat([real,pixel],dim=1)
        return self.model(x)


# --------------------
# Training
# --------------------

train_dataset = Pix2PixDataset(os.path.join(DATASET,"train"))
train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)

G = Generator().to(DEVICE)
D = Discriminator().to(DEVICE)

opt_G = torch.optim.Adam(G.parameters(),lr=LR,betas=(0.5,0.999))
opt_D = torch.optim.Adam(D.parameters(),lr=LR,betas=(0.5,0.999))

bce = nn.BCEWithLogitsLoss()
l1 = nn.L1Loss()

for epoch in range(EPOCHS):

    loop = tqdm(train_loader)

    for real,pixel in loop:

        real = real.to(DEVICE)
        pixel = pixel.to(DEVICE)

        fake = G(real)

        # Train Discriminator

        D_real = D(real,pixel)
        D_fake = D(real,fake.detach())

        loss_D = (
            bce(D_real,torch.ones_like(D_real)) +
            bce(D_fake,torch.zeros_like(D_fake))
        )

        opt_D.zero_grad()
        loss_D.backward()
        opt_D.step()

        # Train Generator

        D_fake = D(real,fake)

        loss_G = (
            bce(D_fake,torch.ones_like(D_fake)) +
            100*l1(fake,pixel)
        )

        opt_G.zero_grad()
        loss_G.backward()
        opt_G.step()

        loop.set_postfix(loss_G=loss_G.item())

print("Training finished")
torch.save(G.state_dict(), "generator.pth")
print("Model saved")