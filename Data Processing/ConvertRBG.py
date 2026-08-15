import os
from PIL import Image

# --------------------
# Input / Output folder
# --------------------
input_dir = r"C:\Users\Jason\Downloads\Newdata\OrignalPix2pix\rgba"
output_dir = r"C:\Users\Jason\Downloads\Newdata\OrignalPix2pix\rgb"

os.makedirs(output_dir, exist_ok=True)

# --------------------
# Loop all images
# --------------------
for filename in os.listdir(input_dir):

    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    path = os.path.join(input_dir, filename)

    img = Image.open(path)

    # 強制轉RGB（會自動drop alpha）
    img_rgb = img.convert("RGB")

    save_path = os.path.join(output_dir, filename)
    img_rgb.save(save_path)

    print(f"Processed: {filename}")

print("✅ All images converted to RGB.")