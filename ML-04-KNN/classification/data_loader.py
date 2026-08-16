"""
data_loader.py
--------------
step:
1. Read the CSV file
2. (If set sample_size) random some a data because my KNN must calculate distance with the poin data 
3.  "pixels" Transform string -> array (number)
4. Use the specified column is label ("gender") 
5. Split the data into sets train/test
6. Standardize the pixel values to mean = 0 and std = 1
"""
 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
 
ck = pd.read_csv(r"C:\Users\USER\Desktop\ML\ML-tase\ML-04-KNN\data-face\age_gender.csv")
def load_data(csv_path, target_column="gender", sample_size=3000,
              test_size=0.2, random_state=42):
    """ 
    paramiter:
        csv_path (str): position the CSV file
        target_column (str): The column name is the label / answer for prediction (gender)
        sample_size (int or None): The number of rows to randomly select. (None = all data)
        test_size (float): Test set proportion 0.2 ( 20%  test, 80%  train)
        random_state (int): A fixed value to ensure reproducible random results. 
 
    return:
        X_train_scaled, X_test_scaled, y_train, y_test
    """
    # 1. read the file CSV
    df = pd.read_csv(csv_path)
    # 2. random some a data to run KNN 
    if sample_size is not None and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


    # 3. trancefrome (string) --> (array) 
    print("trancefrome string --> array .......")
    X = np.stack(
        df["pixels"].apply(lambda p: np.fromstring(p, dtype=float, sep=" ")).values
    )
 
    # 4. Extract the answer (label) from the target_column(gender) column.
    y = df[target_column].values
 
    # 5. set train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 6. Standardize data (Fit on the training data only, then transform both the training and test data.)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    return X_train_scaled, X_test_scaled, y_train, y_test

