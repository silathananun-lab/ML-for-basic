import os

from config import OUTPUT_DIR

from data_loader import (
    load_dataset,
    check_dataset,
    prepare_data
)

from preprocessing import (
    split_dataset,
    create_preprocessor
)

from svm_models import (
    create_svm_models
)

from evaluate import (
    train_models,
    save_accuracy_results,
    plot_accuracy,
    plot_confusion_matrix,
    print_classification_report,
    save_predictions
)


# Main Program

def main():

    # build outputs if none
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )


    print("SVM Classification Program")
    print("")

    # 1. Load Dataset
    df = load_dataset()

    # 2. Check Dataset
    check_dataset(df)

    # 3. Prepare Features and Target
    X, y = prepare_data(df)


    # 4. Split Dataset
    ( X_train,X_test,y_train,y_test) = split_dataset(X,y)


    # 5. Create Preprocessor
    preprocessor = create_preprocessor()

    # 6. Create SVM Models
    models = create_svm_models()

    # 7. Train and Evaluate
    ( results,best_model,best_kernel,best_accuracy,best_predictions) = train_models(models,
                                                                                    preprocessor,
                                                                                    X_train,
                                                                                    X_test,
                                                                                    y_train,
                                                                                    y_test
                                                                                )


    # 8. Save Accuracy Results

    results_df = save_accuracy_results(results,OUTPUT_DIR)


    # 9. Plot Accuracy
    plot_accuracy(results_df, OUTPUT_DIR)


    # 10. Confusion Matrix
    plot_confusion_matrix(y_test,best_predictions,best_kernel,OUTPUT_DIR)


    # 11. Classification Report
    print_classification_report( y_test,best_predictions)

    # 12. Save Predictions
    save_predictions(df,X_test,y_test,best_predictions,OUTPUT_DIR)


    # 13. Final Summary

    print("Experiment Summary")
    print("")

    print( f"Best Kernel: {best_kernel}")

    print(f"Best Accuracy: "f"{best_accuracy:.4f}")

    print("\nAll processes completed successfully!")


if __name__ == "__main__":
    main()