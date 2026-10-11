def standardize_features(X_train, X_test):
    """Normalizes pixel intensity values."""
    X_train_scaled = X_train / 255.0
    X_test_scaled = X_test / 255.0
    return X_train_scaled, X_test_scaled