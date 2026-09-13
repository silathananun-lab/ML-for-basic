import pandas as pd

def load_data():

    train_path = "dataset/sign_mnist_train.csv"
    test_path = "dataset/sign_mnist_test.csv"

    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    print("Train data shape:", train_data.shape)
    print("Test data shape :", test_data.shape)

    # แยก label ออกจาก pixel
    X_train = train_data.drop("label", axis=1)
    y_train = train_data["label"]

    X_test = test_data.drop("label", axis=1)
    y_test = test_data["label"]

    return X_train, X_test, y_train, y_test