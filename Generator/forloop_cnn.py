import os
import torch
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = "cnn_model.pth"
INPUT_DIR = r"C:\Users\Jason\Downloads\Newdata\data\test\real"
OUTPUT_DIR = r"C:\Users\Jason\Downloads\Newdata\CNN_output"

IMG_SIZE_REAL = 128
IMG_SIZE_PIXEL = 32

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------
# Transform
# --------------------

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE_REAL, IMG_SIZE_REAL)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# --------------------
# Model（同 train 一樣）
# --------------------

import torch.nn as nn

class SimpleCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(nn.Conv2d(3,64,4,2,1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU())
        self.enc3 = nn.Sequential(nn.Conv2d(128,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU())

        self.mid = nn.Sequential(nn.Conv2d(256,256,3,1,1), nn.ReLU())

        self.dec1 = nn.Sequential(nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU())

        self.rgb = nn.Conv2d(128,3,3,1,1)
        self.alpha = nn.Conv2d(128,1,3,1,1)

    def forward(self,x):

        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.mid(x)

        x = self.dec1(x)

        rgb = torch.tanh(self.rgb(x))
        alpha = torch.sigmoid(self.alpha(x) * 5)

        return torch.cat([rgb, alpha], dim=1)

# --------------------
# Load model
# --------------------

model = SimpleCNN().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --------------------
# Generate
# --------------------

files = [f for f in os.listdir(INPUT_DIR)
         if f.lower().endswith((".png",".jpg",".jpeg"))]

for name in tqdm(files):

    # load image
    img = Image.open(os.path.join(INPUT_DIR, name)).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)

    output = output.squeeze(0).cpu()

    # --------------------
    # Convert back to image
    # --------------------

    rgb = output[:3]
    alpha = output[3]

    # [-1,1] → [0,1]
    rgb = (rgb + 1) / 2

    # clamp
    rgb = torch.clamp(rgb, 0, 1)
    alpha = torch.clamp(alpha, 0, 1)

    # to numpy
    rgb = (rgb.permute(1,2,0).numpy() * 255).astype("uint8")
    alpha = (alpha.numpy() * 255).astype("uint8")

    # combine RGBA
    rgba = Image.fromarray(rgb)
    alpha_img = Image.fromarray(alpha)

    rgba.putalpha(alpha_img)

    # save
    save_path = os.path.join(OUTPUT_DIR, name.replace(".jpg", ".png"))
    rgba.save(save_path)

print("✅ All images generated in CNN_output_data/")