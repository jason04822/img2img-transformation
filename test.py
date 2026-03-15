import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_SIZE = 128

# same generator structure
import torch.nn as nn


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


# load model

G = Generator().to(DEVICE)
G.load_state_dict(torch.load("generator.pth"))
G.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])


def tensor_to_image(t):

    t = t.detach().cpu()
    t = (t + 1) / 2
    return t.permute(1,2,0).numpy()


# test image
image_path = "test.png"

img = Image.open(image_path).convert("RGB")
input_tensor = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    output = G(input_tensor)

plt.subplot(1,2,1)
plt.title("Input")
plt.imshow(img)

plt.subplot(1,2,2)
plt.title("Generated Pixel")
plt.imshow(tensor_to_image(output[0]))

plt.show()