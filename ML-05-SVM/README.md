# SVM Classification Project

This project uses **Support Vector Machine (SVM)** to predict whether a passenger survived the Titanic accident.

The project compares three different SVM kernels:

* Linear Kernel
* Polynomial Kernel
* RBF Kernel

The goal is to compare the accuracy of each kernel and find the best-performing model.

---

## Project Structure

```text
ML-05-SVM/
│
├── main.py
├── config.py
├── data_loader.py
├── preprocessing.py
├── svm_models.py
├── evaluate.py
├── train.csv
│
├── outputs/
│   ├── accuracy_results.csv
│   ├── svm_kernel_accuracy.png
│   ├── confusion_matrix.png
│   └── predictions.csv
│
└── README.md
```

### File Description

| File               | Description                                   |
| ------------------ | --------------------------------------------- |
| `main.py`          | Runs the complete program                     |
| `config.py`        | Stores project settings and feature names     |
| `data_loader.py`   | Loads and checks the dataset                  |
| `preprocessing.py` | Handles missing values, scaling, and encoding |
| `svm_models.py`    | Creates the three SVM models                  |
| `evaluate.py`      | Trains models and evaluates the results       |
| `train.csv`        | Dataset used for training and testing         |
| `outputs/`         | Stores results and graphs                     |

---

## Dataset

This project uses a **Titanic dataset**.

The target variable is:

```text
Survived
```

The model uses the following features:

* `Pclass` - Passenger class
* `Sex` - Passenger gender
* `Age` - Passenger age
* `SibSp` - Number of siblings or spouses
* `Parch` - Number of parents or children
* `Fare` - Ticket fare
* `Embarked` - Port of embarkation

---

## Data Preprocessing

Before training the SVM models, the data is prepared using the following steps:

### 1. Missing Values

Numeric columns use the **median** to fill missing values.

Categorical columns use the **most frequent value**.

### 2. Standardization

Numeric features are standardized using:

```text
StandardScaler
```

This helps the SVM model work better with features that have different scales.

### 3. One-Hot Encoding

Categorical features such as `Sex` and `Embarked` are converted into numerical values using:

```text
OneHotEncoder
```

---

## SVM Models

The project compares three SVM kernels.

### Linear Kernel

```text
kernel = "linear"
C = 1
```

The Linear Kernel is suitable when the classes can be separated using a relatively simple boundary.

### Polynomial Kernel

```text
kernel = "poly"
C = 1
degree = 3
```

The Polynomial Kernel can create a more complex decision boundary.

### RBF Kernel

```text
kernel = "rbf"
C = 1
gamma = "scale"
```

The RBF Kernel can handle more complex relationships between the features.

---

## Train and Test

The dataset is divided into:

```text
80% Training Data
20% Testing Data
```

The model is trained using the training data and evaluated using the testing data.

---

## Evaluation

The project evaluates the models using:

### Accuracy

Accuracy shows the percentage of predictions that are correct.

The results of all three kernels are compared to find the best model.

### Confusion Matrix

The confusion matrix shows:

* Correctly predicted survivors
* Correctly predicted non-survivors
* Incorrect predictions

### Classification Report

The classification report provides:

* Precision
* Recall
* F1-score
* Support

---

## Output

After running the program, the following files are created in the `outputs` folder:

### Accuracy Results

```text
accuracy_results.csv
```

Contains the accuracy score of each SVM kernel.

### Accuracy Graph

```text
svm_kernel_accuracy.png
```

Shows a comparison of the accuracy of the three SVM kernels.

### Confusion Matrix

```text
confusion_matrix.png
```

Shows the prediction performance of the best SVM model.

### Predictions

```text
predictions.csv
```

Contains the actual and predicted results for the test data.

Example:

```text
PassengerId,Actual,Predicted
1,0,0
2,1,1
3,1,0
4,1,1
```

---

## Installation

Make sure Python is installed on your computer.

Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

---

## How to Run

Open a terminal in the project folder:

```bash
cd ML-05-SVM
```

Then run:

```bash
python main.py
```

The program will:

1. Load the dataset
2. Check the data
3. Split the data into training and testing sets
4. Preprocess the data
5. Train three SVM models
6. Calculate the accuracy of each model
7. Find the best kernel
8. Create a confusion matrix
9. Create a classification report
10. Save the predictions

---

## Example Output

```text
==============================
SVM Training
==============================

Training Linear Kernel...
Linear Accuracy = 0.8156

Training Polynomial Kernel...
Polynomial Accuracy = 0.8212

Training RBF Kernel...
RBF Accuracy = 0.8268

==============================
Accuracy Comparison
==============================

       Kernel  Accuracy
0      Linear    0.8156
1  Polynomial    0.8212
2         RBF    0.8268

Best Kernel: RBF
Best Accuracy: 0.8268
```

> The accuracy values may be different when the program is run with a different dataset or configuration.

---

## Technologies Used

* Python
* Pandas
* Scikit-learn
* Matplotlib
* Support Vector Machine (SVM)

---

## Conclusion

This project demonstrates how SVM can be used for a classification problem.

Three different kernels are compared:

* Linear
* Polynomial
* RBF

The kernel with the highest accuracy is selected as the best model.

The project also demonstrates basic machine learning steps such as data preprocessing, model training, evaluation, and prediction.
