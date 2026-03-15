import os
import random
from PIL import Image

REAL_DATASET = r"C:\Users\Jason\Downloads\Animal_FYP\REAL"
PIXEL_DATASET = r"C:\Users\Jason\Downloads\Animal_FYP\PIXEL"
OUTPUT = r"C:\Users\Jason\Downloads\Animal_FYP\dataset_ModelA"

IMG_SIZE = 128

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def resize_image(path):
    img = Image.open(path).convert("RGBA")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    return img


def create_folders():

    for split in ["train", "val", "test"]:
        os.makedirs(f"{OUTPUT}/{split}/real", exist_ok=True)
        os.makedirs(f"{OUTPUT}/{split}/pixel", exist_ok=True)


def process_class(cls, index_counter):

    real_folder = os.path.join(REAL_DATASET, cls)
    pixel_folder = os.path.join(PIXEL_DATASET, cls)

    real_imgs = sorted(os.listdir(real_folder))
    pixel_imgs = sorted(os.listdir(pixel_folder))

    n = min(len(real_imgs), len(pixel_imgs))

    pairs = []

    for i in range(n):

        real_path = os.path.join(real_folder, real_imgs[i])
        pixel_path = os.path.join(pixel_folder, pixel_imgs[i])

        pairs.append((real_path, pixel_path))

    random.shuffle(pairs)

    train_end = int(n * TRAIN_RATIO)
    val_end = int(n * (TRAIN_RATIO + VAL_RATIO))

    splits = {
        "train": pairs[:train_end],
        "val": pairs[train_end:val_end],
        "test": pairs[val_end:]
    }

    for split in splits:

        for real_path, pixel_path in splits[split]:

            index_counter += 1
            name = f"{index_counter:05d}.png"

            real_img = resize_image(real_path)
            pixel_img = resize_image(pixel_path)

            real_img.save(f"{OUTPUT}/{split}/real/{name}")
            pixel_img.save(f"{OUTPUT}/{split}/pixel/{name}")

    return index_counter


def main():

    create_folders()

    classes = os.listdir(REAL_DATASET)

    index_counter = 0

    for cls in classes:

        print(f"Processing {cls}")

        index_counter = process_class(cls, index_counter)

    print("✅ Dataset prepared successfully")


if __name__ == "__main__":
    main()