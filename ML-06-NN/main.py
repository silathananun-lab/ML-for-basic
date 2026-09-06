import os
import json
from pathlib import Path

import numpy as np
from data_loader import load_dataset
from split_data import split_dataset
from preprocessing import standardize_data
from nn_model import create_model

from evaluate import (
    evaluate_model,
    plot_training,
    plot_prediction
)

PROJECT_DIR = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_DIR / "data" / "age_gender.csv"
OUTPUT_DIR = PROJECT_DIR / "outputs"
CONFIG = "A"
EPOCHS = 50


# 1. Load Dataset
X, y = load_dataset(DATASET_PATH)

# 2. Split Dataset
( X_train, X_val,X_test,y_train,y_val,y_test) = split_dataset(X,y)

# 3. Standardization
(X_train,X_val,X_test,scaler) = standardize_data(X_train,X_val,X_test)

# Save Data
OUTPUT_DIR.mkdir(exist_ok=True)

np.save(
    OUTPUT_DIR / "X_train.npy",
    X_train
)

np.save(
    OUTPUT_DIR / "X_val.npy",
    X_val
)

np.save(
    OUTPUT_DIR / "X_test.npy",
    X_test
)

np.save(
    OUTPUT_DIR / "y_train.npy",
    y_train
)

np.save(
    OUTPUT_DIR / "y_val.npy",
    y_val
)

np.save(
    OUTPUT_DIR / "y_test.npy",
    y_test
)

# 4. Build Neural Network
model = create_model(
    input_size=X_train.shape[1],
    config=CONFIG
)

model.summary()

# 5. Train
history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val,y_val),
    epochs=EPOCHS,
    batch_size=64,
    verbose=1
)

# 6. Save Model
model.save(OUTPUT_DIR / "nn_model.h5")

# 7. Evaluation
prediction, mae, rmse, r2 = evaluate_model(model,X_test, y_test)

# 8. Save Predictions
import pandas as pd

result = pd.DataFrame({"Actual_Age": y_test,"Predicted_Age": prediction,"Absolute_Error":
        np.abs(
            y_test - prediction
        )

})

result.to_csv(
    OUTPUT_DIR / "predictions.csv",
    index=False
)

# 9. Plot
plot_training(
    history
)

plot_prediction(
    y_test,
    prediction
)

# 10. Print Result
print("\n==============================")

print("Final Result")

print("==============================")

print(
    f"MAE  : {mae:.2f} years"
)

print(
    f"RMSE : {rmse:.2f} years"
)

print(
    f"R2   : {r2:.4f}"
)