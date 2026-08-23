from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from config import (
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
    TEST_SIZE,
    RANDOM_STATE
)



# Split Dataset

def split_dataset(X, y):

    #  Training 80%  Testing 20%
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    print("\n==============================")
    print("Dataset Split")
    print("==============================")

    print("Training data:", X_train.shape)
    print("Testing data :", X_test.shape)

    return X_train, X_test, y_train, y_test



# Create Preprocessor


def create_preprocessor():

    # Preprocessor data
    # 1. add Missing Value from Median
    # 2. Standardization
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # จัดการข้อมูลประเภทข้อความ
    # 1.  Missing Value frome mean value 
    # 2.  One-Hot Encoding
    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore")
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_transformer,
                NUMERIC_FEATURES
            ),
            (
                "categorical",
                categorical_transformer,
                CATEGORICAL_FEATURES
            )
        ]
    )

    return preprocessor