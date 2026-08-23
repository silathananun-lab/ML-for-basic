from sklearn.svm import SVC



# Create SVM Models
def create_svm_models():

    models = {

        "Linear": SVC(
            kernel="linear",
            C=1
        ),

        "Polynomial": SVC(
            kernel="poly",
            C=1,
            degree=3
        ),

        "RBF": SVC(
            kernel="rbf",
            C=1,
            gamma="scale"
        )
    }

    return models