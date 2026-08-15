import os
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
import numpy as np

# --------------------
# Config
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 128

# --------------------
# Generator（同 train 完全一致）
# --------------------
class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc0 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.down1 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.down5 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(1024, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.up5 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

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

# --------------------
# Load model
# --------------------
G = Generator().to(DEVICE)
G.load_state_dict(torch.load("generator_u5_128.pth", map_location=DEVICE))
G.eval()

# --------------------
# Transform（🔥 你之前 missing）
# --------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# --------------------
# Input folder
# --------------------
input_dir = r"C:\Users\Jason\Downloads\Newdata\data\test\real"

# --------------------
# Output folders
# --------------------
save_root = r"C:\Users\Jason\Downloads\Newdata\OrignalPix2pix"

rgb_dir   = os.path.join(save_root, "rgb")
rgba_dir  = os.path.join(save_root, "rgba")
alpha_dir = os.path.join(save_root, "alpha")

os.makedirs(rgb_dir, exist_ok=True)
os.makedirs(rgba_dir, exist_ok=True)
os.makedirs(alpha_dir, exist_ok=True)

# --------------------
# Loop all images
# --------------------
for filename in os.listdir(input_dir):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    image_path = os.path.join(input_dir, filename)

    # load image
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    # inference
    with torch.no_grad():
        out = G(x)[0]

    out = out.detach().cpu()

    # split channels
    rgb = (out[:3] + 1) / 2
    alpha = out[3:4]

    rgb_np = rgb.permute(1,2,0).numpy()
    alpha_np = alpha.permute(1,2,0).numpy()

    # RGBA（🔥 真4 channel）
    rgba = np.concatenate([rgb_np, alpha_np], axis=2)

    name = os.path.splitext(filename)[0]

    # --------------------
    # Save RGBA
    # --------------------
    Image.fromarray((rgba * 255).astype('uint8'), mode="RGBA").save(
        os.path.join(rgba_dir, f"{name}.png")
    )

    print(f"Processed: {filename}")

print("✅ All images done.")