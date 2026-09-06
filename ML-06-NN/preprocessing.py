from sklearn.preprocessing import StandardScaler


def standardize_data(X_train, X_val, X_test):

    print("")
    print("*standardize_data*")
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)

    X_val = scaler.transform(X_val)

    X_test = scaler.transform(X_test)
    print("*finished*")
    return X_train, X_val, X_test, scaler