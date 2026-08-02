# Age prediction
Age prediction is Analyze data to estimate or assess a person's age.
## Libraries used
```base
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau
```

## Preprocess Data
```base
X = []
for p in df["pixels"]:
    img = np.array(p.split(), dtype=np.float32)
    img = img.reshape(48, 48)
    X.append(img)
```
* X = []: build the list for keep the picture data.
* for p in df["pixels"]:: loop for read data in colum pixels. 
* X.append(img): Save the processed images into list X.

```base
X = np.array(X)
X = X / 255.0                  # Normalize pixels value between 0 - 1
X = X.reshape(-1, 48, 48, 1)   # Reshape for into CNN (48x48x1)
y = df["age"].values
```
* X = np.array(X): Converts the list of all images into a NumPy array.
* X = X / 255.0 : Normalizes pixel values from the range $[0, 255]$ to $[0.0, 1.0]$ to facilitate efficient model training.
* y = df["age"].values: Extracts the age values from the age column into variable y to serve as the target output.

```base
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
* Split data Train / Test (80% / 20%)

## CNN (Model Architecture)
```base
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Fully Connected
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1)  
])
```
* Start to build the structure model 
* Block 1 : Extracting low-level features.
* Block 2 :Combine Block 1 features to capture complex patterns (eyes, nose, lips)
* Block 3 :Combine all patterns into a full facial structure (face shape, wrinkles, biological traits)
* Flatten 2D/3D feature maps to a 1D vector to compute the final predicted age

## Model Compilation & Callbacks
### Compile Model
```base
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()
```
* Use Adam Optimizer to learning rate of 0.001.
* Use mse 
* Use mae 
* `model.summary()` Display model summary architecture and parameter count

### Callback healp to  Learning Rate auto
```base
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)
```

## Training
```base
print("\nStarting training...")
history = model.fit(
    X_train,
    y_train,
    epochs=35,           
    batch_size=64,
    validation_split=0.2,
    callbacks=[reduce_lr]
)
```