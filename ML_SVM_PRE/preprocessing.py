from sklearn.preprocessing import StandardScaler

def preprocess_data(X_train, X_test):

    scaler = StandardScaler()

    # Fit เฉพาะข้อมูล Train
    X_train_scaled = scaler.fit_transform(X_train)
    
    # ใช้ scaler เดียวกันกับ Test
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler