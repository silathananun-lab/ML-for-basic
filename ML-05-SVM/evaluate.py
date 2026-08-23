import os

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)



# Train and Evaluate Models


def train_models(
    models,
    preprocessor,
    X_train,
    X_test,
    y_train,
    y_test
):

    results = []

    best_model = None
    best_kernel = None
    best_accuracy = 0
    best_predictions = None


    print("SVM Training")
    print("")

    for kernel_name, svm_model in models.items():

        print(
            f"\nTraining {kernel_name} Kernel..."
        )

        # include Preprocessing and SVM 
        model = Pipeline(
            steps=[
                (
                    "preprocessing",
                    preprocessor
                ),
                (
                    "svm",
                    svm_model
                )
            ]
        )

        # Train Model
        model.fit(
            X_train,
            y_train
        )

        # Predict Test Data
        predictions = model.predict(
            X_test
        )

        # calculate Accuracy
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        print(
            f"{kernel_name} Accuracy = "
            f"{accuracy:.4f}"
        )

        # keep the results
        results.append({
            "Kernel": kernel_name,
            "Accuracy": accuracy
        })

        # check the base Model 
        if accuracy > best_accuracy:

            best_accuracy = accuracy

            best_kernel = kernel_name

            best_model = model

            best_predictions = predictions

    return (
        results,
        best_model,
        best_kernel,
        best_accuracy,
        best_predictions
    )



# Save Accuracy Results

def save_accuracy_results(
    results,
    output_dir
):

    results_df = pd.DataFrame(results)


    print("Accuracy Comparison")
    print("")

    print(results_df)

    best_row = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]

    print(
        f"\nBest Kernel: "
        f"{best_row['Kernel']}"
    )

    print(
        f"Best Accuracy: "
        f"{best_row['Accuracy']:.4f}"
    )

    # save Accuracy --> CSV
    accuracy_path = os.path.join(
        output_dir,
        "accuracy_results.csv"
    )

    results_df.to_csv(
        accuracy_path,
        index=False
    )

    print(
        f"\nAccuracy results saved to: "
        f"{accuracy_path}"
    )

    return results_df



# Plot Accuracy

def plot_accuracy(
    results_df,
    output_dir
):

    plt.figure(figsize=(8, 5))

    plt.bar(
        results_df["Kernel"],
        results_df["Accuracy"]
    )

    plt.title(
        "SVM Kernel Accuracy Comparison"
    )

    plt.xlabel("Kernel")

    plt.ylabel("Accuracy")

    plt.ylim(0, 1)

    # display Accuracy on the ghrap
    for i, value in enumerate(
        results_df["Accuracy"]
    ):

        plt.text(
            i,
            value + 0.02,
            f"{value:.3f}",
            ha="center"
        )

    plt.tight_layout()

    accuracy_graph_path = os.path.join(
        output_dir,
        "svm_kernel_accuracy.png"
    )

    plt.savefig(
        accuracy_graph_path
    )

    plt.show()

    print(
        f"Accuracy graph saved to: "
        f"{accuracy_graph_path}"
    )



# Plot Confusion Matrix


def plot_confusion_matrix(
    y_test,
    predictions,
    best_kernel,
    output_dir
):

    # build Confusion Matrix
    cm = confusion_matrix(
        y_test,
        predictions
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[
            "Not Survived",
            "Survived"
        ]
    )

    disp.plot()

    plt.title(
        f"Confusion Matrix - {best_kernel}"
    )

    plt.tight_layout()

    confusion_path = os.path.join(
        output_dir,
        "confusion_matrix.png"
    )

    plt.savefig(
        confusion_path
    )

    plt.show()

    print(
        f"Confusion matrix saved to: "
        f"{confusion_path}"
    )


# Classification Report


def print_classification_report(
    y_test,
    predictions
):

    print("\n==============================")
    print("Classification Report")
    print("==============================")

    print(
        classification_report(
            y_test,
            predictions
        )
    )



# Save Predictions


def save_predictions(
    df,
    X_test,
    y_test,
    predictions,
    output_dir
):

    prediction_df = pd.DataFrame({

        "PassengerId":
            df.loc[
                X_test.index,
                "PassengerId"
            ],

        "Actual":
            y_test,

        "Predicted":
            predictions
    })


    prediction_df = prediction_df.sort_values(
        "PassengerId"
    )

    prediction_path = os.path.join(
        output_dir,
        "predictions.csv"
    )

    prediction_df.to_csv(
        prediction_path,
        index=False
    )

    print(
        f"\nPredictions saved to: "
        f"{prediction_path}"
    )


    print("Prediction Examples")
    print("")

    print(
        prediction_df.head(10)
    )

    return prediction_df