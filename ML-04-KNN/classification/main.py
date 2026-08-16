
 
import os
import pandas as pd
 
from data_loader import load_data
from KNN import knn_predict
from Evaluate import calculate_accuracy, plot_k_curve, plot_confusion_matrix
 

DATA_PATH = "../data-face/age_gender.csv"  # modify part for position dataset
TARGET_COLUMN = "gender"   # select data type (gender 2 class) or (ethnicity 5 class)  
SAMPLE_SIZE = 3000         # Number of images used
OUTPUT_DIR = "outputs"
K_VALUES = [3, 5, 7]       # K valude 

 
 
def main():
    # Create the outputs folder if it doesn't already exist.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    print("Download data and Preparing data .....")
    X_train, X_test, y_train, y_test = load_data(
        DATA_PATH, target_column=TARGET_COLUMN, sample_size=SAMPLE_SIZE
    )
    print(f"number of data train: {len(X_train)} | number of data tase: {len(X_test)}")
 
    accuracies = []
    best_k = None
    best_acc = -1
    best_pred = None
 
    # Train and evaluate the model for each value of k.
    for k in K_VALUES:
        print(f"\nTraining the model with k = {k} ... (just a moment)")
        y_pred = knn_predict(X_train, y_train, X_test, k)
        for i in range(0, k):
            print (f" y_test {y_test[i]} , ")
        print (f"\n")
        for i in range(0, k):
            print (f" y_pred {y_pred[i]} , ")
        acc = calculate_accuracy(y_test, y_pred)
        accuracies.append(acc)
        print(f"  -> accuracy when k = {k}: {acc:.4f}")
 
        # Store the results for the best k value to use for the confusion matrix and predictions.csv
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_pred = y_pred
 
    # Save a graph comparing the accuracy of each k value
    plot_k_curve(K_VALUES, accuracies, os.path.join(OUTPUT_DIR, "01_k_curve.png"))
 
    # Save a graph comparing the accuracy of each k value
    plot_confusion_matrix(
        y_test, best_pred, os.path.join(OUTPUT_DIR, "02_confusion_matrix.png")
    )
 
    # Save a table comparing the actual labels with the predicted labels (for the best k only)
    results_df = pd.DataFrame({
        "actual": y_test,
        "predicted": best_pred
    })
    results_df.to_csv(os.path.join(OUTPUT_DIR, "predictions.csv"), index=False)
 
    # Save a table comparing the actual labels with the predicted labels (for the best k only)
    print("\n===== Results =====")
    for k, acc in zip(K_VALUES, accuracies):
        print(f"k = {k}: accuracy = {acc:.4f}")
    print(f"\nThe best k value is k = {best_k} (accuracy = {best_acc:.4f})")
    print(f"Save all results in the folder '{OUTPUT_DIR}/' Done.")
 
 
if __name__ == "__main__":
    main()
 