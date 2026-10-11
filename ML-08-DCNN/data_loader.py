import pandas as pd
import numpy as np

def load_face_data(csv_path):
    """Loads dataset and parses space-separated pixels into 48x48 images."""
    df = pd.read_csv(csv_path)
    pixels = df['pixels'].tolist()
    
    X = np.array([np.fromstring(pixel, dtype=float, sep=' ') for pixel in pixels])
    X = X.reshape(-1, 48, 48, 1)
    
    # Example target: Gender classification (0 or 1)
    y = df['gender'].values
    return X, y