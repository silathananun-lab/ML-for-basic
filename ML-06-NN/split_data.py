from sklearn.model_selection import train_test_split


def split_dataset(X, y):

    print("")
    print("*split_dataset*")
    # split Train 70% and Temp 30%
    X_train, X_temp, y_train, y_temp = train_test_split(X,y,test_size=0.30,random_state=42)

    # Temp half for Validation and half for Test
    # Validation 15%
    # Test 15%

    X_val, X_test, y_val, y_test = train_test_split( X_temp,y_temp,test_size=0.50,random_state=42)

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)
    print("*finished*")
    return X_train, X_val, X_test, y_train, y_val, y_test