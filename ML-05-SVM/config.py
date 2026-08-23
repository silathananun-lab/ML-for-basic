
# Configuration

DATA_PATH = "train.csv"

OUTPUT_DIR = "outputs"

# Features 
FEATURES = [
    "Pclass",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Fare",
    "Embarked"
]

# Target 
TARGET = "Survived"

# Features is numeric
NUMERIC_FEATURES = [
    "Pclass",
    "Age",
    "SibSp",
    "Parch",
    "Fare"
]

# Features is text
CATEGORICAL_FEATURES = [
    "Sex",
    "Embarked"
]

#  Test 20% train 80%
TEST_SIZE = 0.2

# random value beging
RANDOM_STATE = 42