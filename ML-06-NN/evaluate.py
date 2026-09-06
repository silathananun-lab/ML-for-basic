import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"


def evaluate_model(model, X_test, y_test):

    prediction = model.predict(
        X_test,
        verbose=0
    ).flatten()

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    r2 = r2_score(
        y_test,
        prediction
    )

    print("\nEvaluation")
    print("-----------------------")

    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)

    return prediction, mae, rmse, r2


def plot_training(history):

    OUTPUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history["loss"],
        label="Training Loss"
    )

    plt.plot(
        history.history["val_loss"],
        label="Validation Loss"
    )

    plt.xlabel("Epoch")
    plt.ylabel("MAE Loss")

    plt.title(
        "Training vs Validation Loss"
    )

    plt.legend()

    plt.grid()

    plt.savefig(
        OUTPUT_DIR / "training_history.png"
    )

    plt.show()


def plot_prediction(y_test, prediction):

    OUTPUT_DIR.mkdir(exist_ok=True)

    plt.figure(figsize=(7, 7))

    plt.scatter(
        y_test,
        prediction,
        alpha=0.3
    )

    # line for the perfect prediction
    minimum = min(
        y_test.min(),
        prediction.min()
    )

    maximum = max(
        y_test.max(),
        prediction.max()
    )

    plt.plot(
        [minimum, maximum],
        [minimum, maximum],
        linestyle="--"
    )

    plt.xlabel("Actual Age")

    plt.ylabel("Predicted Age")

    plt.title(
        "Actual Age vs Predicted Age"
    )

    plt.grid()

    plt.savefig(
        OUTPUT_DIR / "prediction_result.png"
    )

    plt.show()