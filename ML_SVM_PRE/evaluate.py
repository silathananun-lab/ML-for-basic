from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt

LABELS = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "I",
    "K", "L", "M", "N", "O",
    "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y"
]


def evaluate_model(model, X_test, y_test):
    # Predict
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("+++++++ SVM Evaluation ++++++++")

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    report = classification_report(
        y_test,
        y_pred,
        target_names=LABELS,
        zero_division=0,
        output_dict=True
    )

    # %
    print(
        f"{'Class':<10}"
        f"{'Precision':>12}"
        f"{'Recall':>12}"
        f"{'F1-Score':>12}"
        f"{'Support':>10}"
    )

    print("-" * 56)

    for label in LABELS:

        precision = report[label]["precision"] * 100
        recall = report[label]["recall"] * 100
        f1 = report[label]["f1-score"] * 100
        support = int(report[label]["support"])

        print(
            f"{label:<10}"
            f"{precision:>11.2f}%"
            f"{recall:>11.2f}%"
            f"{f1:>11.2f}%"
            f"{support:>10}"
        )

    print("-" * 56)

    # Macro Average
    print(
        f"{'Macro Avg':<10}"
        f"{report['macro avg']['precision'] * 100:>11.2f}%"
        f"{report['macro avg']['recall'] * 100:>11.2f}%"
        f"{report['macro avg']['f1-score'] * 100:>11.2f}%"
        f"{int(report['macro avg']['support']):>10}"
    )

    # Weighted Average
    print(
        f"{'Weighted Avg':<10}"
        f"{report['weighted avg']['precision'] * 100:>11.2f}%"
        f"{report['weighted avg']['recall'] * 100:>11.2f}%"
        f"{report['weighted avg']['f1-score'] * 100:>11.2f}%"
        f"{int(report['weighted avg']['support']):>10}"
    )

    return y_pred


def show_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    plt.imshow(cm)
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.xticks(range(len(LABELS)),LABELS)
    plt.yticks(range(len(LABELS)),LABELS)
    plt.colorbar()
    plt.tight_layout()
    plt.show()