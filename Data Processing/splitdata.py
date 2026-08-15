import os
import random
from PIL import Image

# ====== PATH ======
REAL_DATASET = r"C:\Users\Jason\Downloads\Newdata\dataset\real"
PIXEL_DATASET = r"C:\Users\Jason\Downloads\Newdata\dataset\pixel"
OUTPUT = r"C:\Users\Jason\Downloads\Newdata\data"

IMG_SIZE = 128

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


# ====== IMAGE PROCESS ======
def resize_image(path):
    img = Image.open(path).convert("RGBA")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return img


# ====== CREATE OUTPUT FOLDER ======
def create_folders():
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUTPUT, split, "real"), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT, split, "pixel"), exist_ok=True)


# ====== GET IMAGE FILES ONLY ======
def get_image_files(folder):
    files = os.listdir(folder)
    return sorted([
        f for f in files
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])


# ====== MAIN ======
def main():

    print("🔍 Checking dataset...")

    create_folders()

    real_imgs = get_image_files(REAL_DATASET)
    pixel_imgs = get_image_files(PIXEL_DATASET)

    print(f"REAL count: {len(real_imgs)}")
    print(f"PIXEL count: {len(pixel_imgs)}")

    # ====== MATCH BY FILENAME ======
    common = sorted(list(set(real_imgs) & set(pixel_imgs)))

    if len(common) == 0:
        print("❌ ERROR: No matching filenames between real & pixel")
        return

    print(f"✅ Matched pairs: {len(common)}")

    # ====== CREATE PAIRS ======
    pairs = []
    for name in common:
        real_path = os.path.join(REAL_DATASET, name)
        pixel_path = os.path.join(PIXEL_DATASET, name)
        pairs.append((real_path, pixel_path))

    # ====== SHUFFLE ======
    random.shuffle(pairs)

    # ====== SPLIT ======
    n = len(pairs)
    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:]
    }

    print(f"Train: {len(splits['train'])}")
    print(f"Val: {len(splits['val'])}")
    print(f"Test: {len(splits['test'])}")

    # ====== SAVE ======
    index_counter = 0

    for split in splits:
        print(f"\n📂 Processing {split}...")

        for real_path, pixel_path in splits[split]:

            index_counter += 1
            name = f"{index_counter:05d}.png"

            try:
                real_img = resize_image(real_path)
                pixel_img = resize_image(pixel_path)

                real_img.save(os.path.join(OUTPUT, split, "real", name))
                pixel_img.save(os.path.join(OUTPUT, split, "pixel", name))

            except Exception as e:
                print(f"❌ Error processing {real_path}: {e}")

    print("\n✅ Dataset prepared successfully!")


# ====== RUN ======
if __name__ == "__main__":
    main()