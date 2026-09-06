import pandas as pd
import numpy as np


def load_dataset(csv_path):

    print("Loading dataset...")

    df = pd.read_csv(csv_path)

    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())

    #delete the data that has no age or pixels values
    df = df.dropna(subset=["age", "pixels"])

    X = []
    y = []

    skipped = 0

    for _, row in df.iterrows():

        try:

            pixels = np.fromstring(row["pixels"], sep=" ")

            # the image must have 48x48 = 2304 pixels
            if len(pixels) != 48 * 48:
                skipped += 1
                continue

            X.append(pixels)
            y.append(float(row["age"]))

        except Exception:
            skipped += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print("Valid samples:", len(X))
    print("Skipped:", skipped)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("*finished*")
    return X, y