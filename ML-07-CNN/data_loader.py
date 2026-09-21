import os
import cv2
import numpy as np
import pandas as pd

from preprocessing import preprocess_image


def load_data(data_path, img_size=100, max_per_class=None):

    if not os.path.isfile(data_path):
        raise FileNotFoundError(f"Dataset not found: {os.path.abspath(data_path)}")

    print(f"Loading CSV: {os.path.abspath(data_path)}")

    df = pd.read_csv(data_path)

    print("Dataset columns:", list(df.columns))
    print("Total rows:", len(df))

    images = []
    labels = []

    # Dataset: 0 = Male, 1 = Female
    class_names = ["male", "female"]

    # แยกจำนวนสูงสุดต่อ class
    for label in [0, 1]:

        class_data = df[df["gender"] == label]

        if max_per_class is not None:
            class_data = class_data.head(max_per_class)

        loaded = 0
        skipped = 0

        for _, row in class_data.iterrows():

            try:
                # pixels เป็นตัวเลข 0-255 คั่นด้วยช่องว่าง
                pixels = np.fromstring(row["pixels"], dtype=np.uint8, sep=" ")

                # Dataset เดิมเป็น 48x48
                image = pixels.reshape(48, 48)

                # Resize 48x48 -> img_size x img_size
                image = cv2.resize(
                    image,
                    (img_size, img_size)
                )

                # preprocessing
                image = preprocess_image(image, img_size)

                if image is None:
                    skipped += 1
                    continue

                images.append(image)
                labels.append(label)

                loaded += 1

            except Exception:
                skipped += 1

        print(
            f"Loaded class {class_names[label]}: "
            f"{loaded} images ({skipped} skipped)"
        )

    if len(images) == 0:
        raise ValueError("No images were loaded from the dataset.")

    images = np.array(images)
    labels = np.array(labels)

    print("Final image shape:", images.shape)
    print("Final label shape:", labels.shape)

    return images, labels, class_names