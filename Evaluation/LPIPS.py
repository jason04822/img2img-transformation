import os
import torch
import lpips
from PIL import Image
from torchvision import transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

gt_dir = r"C:\Users\Jason\Downloads\Newdata\groundturth_rgb"
gen_dir = r"C:\Users\Jason\Downloads\Newdata\OrignalPix2pix\tenth train\rgb"

# LPIPS model
loss_fn = lpips.LPIPS(net='alex').to(device)

# transform（LPIPS requires -1~1）
transform = transforms.Compose([
    transforms.Resize((32,32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

# Loop
scores = []

for filename in os.listdir(gt_dir):

    if not filename.endswith(".png"):
        continue

    gt_path = os.path.join(gt_dir, filename)
    gen_path = os.path.join(gen_dir, filename)

    # skip if no pair
    if not os.path.exists(gen_path):
        print(f"Missing: {filename}")
        continue

    img_gt = Image.open(gt_path).convert("RGB")
    img_gen = Image.open(gen_path).convert("RGB")

    t_gt = transform(img_gt).unsqueeze(0).to(device)
    t_gen = transform(img_gen).unsqueeze(0).to(device)

    with torch.no_grad():
        score = loss_fn(t_gt, t_gen)

    score_val = score.item()
    scores.append(score_val)

    print(f"{filename}: {score_val:.4f}")

# Final result
avg_score = sum(scores) / len(scores)

print("\n====================")
print(f"LPIPS Average: {avg_score:.4f}")
print("====================")