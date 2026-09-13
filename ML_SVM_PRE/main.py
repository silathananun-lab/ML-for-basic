from data_loader import load_data
from preprocessing import preprocess_data
from SVM_model import create_model, train_model
from evaluate import evaluate_model, show_confusion_matrix


def main():
    print("+++++++ Sign Language SVM +++++++")

    # 1. Load Dataset
    print("\n[1] Loading dataset...")
    X_train, X_test, y_train, y_test = load_data()

    # 2. Preprocessing
    print("\n[2] Preprocessing...")
    X_train, X_test, scaler = preprocess_data(X_train,X_test)

    # 3. Create SVM
    print("\n[3] Creating SVM model...")
    model = create_model()

    # 4. Train
    print("\n[4] Training model...")
    model = train_model(model,X_train,y_train)

    # 5. Evaluate
    print("\n[5] Evaluating model...")
    y_pred = evaluate_model(model,X_test,y_test)

    # 6. Confusion Matrix
    print("\n[6] Showing confusion matrix...")
    show_confusion_matrix(y_test,y_pred)


if __name__ == "__main__":
    main()