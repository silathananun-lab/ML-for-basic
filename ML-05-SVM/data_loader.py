import pandas as pd

from config import (
    DATA_PATH,
    FEATURES,
    TARGET
)



# Load Dataset
def load_dataset():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nDataset shape:")
    print(df.shape)

    return df



# Check Dataset

def check_dataset(df):

    print("Dataset Information")
    print("")

    # check Missing Values
    print("\nMissing values:")
    print(df.isnull().sum())

    # check number of the data of each Class
    print("\nSurvived class distribution:")
    print(df[TARGET].value_counts())



# Prepare Features and Target

def prepare_data(df):

    # selec Features
    X = df[FEATURES]

    # selec Target
    y = df[TARGET]

    return X, y