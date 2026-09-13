from sklearn.svm import SVC

def create_model():
    model = SVC(kernel="rbf",C=10,gamma="scale")
    return model


def train_model(model, X_train, y_train):
    print("Training SVM...")
    model.fit(X_train, y_train)
    print("Training complete!")
    return model