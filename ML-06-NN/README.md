# Age Prediction using Neural Network

This project uses a Neural Network (NN) to predict a person's age from a face image.

The project uses the Age, Gender and Ethnicity Face Data dataset. The goal is to train a Neural Network and compare different model configurations and numbers of epochs to find a suitable model.

---

## Project Structure

```text
ML-06-NN/
│
├── Age_Gender_Ethnicity/
│   └── age_gender.csv
│
├── regression/
│   ├── main.py
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── split_data.py
│   ├── nn_model.py
│   ├── evaluate.py
│   ├── experiment.py
│   └── test_nn.py
│
├── outputs/
│   ├── X_train.npy
│   ├── X_val.npy
│   ├── X_test.npy
│   ├── y_train.npy
│   ├── y_val.npy
│   ├── y_test.npy
│   ├── nn_model.keras
│   ├── predictions.csv
│   ├── experiment_results.csv
│   ├── training_history.png
│   └── predictions.png
│
└── requirements.txt
```

### File Description

File | Description
--- | ---
`main.py` | Runs the complete Neural Network project
`data_loader.py` | Loads the dataset and prepares the initial data
`preprocessing.py` | Converts pixel data and performs data preprocessing
`split_data.py` | Splits the dataset into training, validation, and testing sets
`nn_model.py` | Creates and configures the Neural Network model
`evaluate.py` | Evaluates the model using regression metrics
`experiment.py` | Runs experiments with different model configurations and epochs
`test_nn.py` | Tests the trained Neural Network and makes predictions
`age_gender.csv` | Face dataset containing age, gender, ethnicity, and pixel data
`outputs/` | Stores processed data, trained model, predictions, and graphs
`requirements.txt` | Contains the Python libraries required for the project

---

## Dataset

This project uses the **Age, Gender and Ethnicity Face Data** dataset.

Dataset source:

https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv

The dataset contains:

- `Age` - Age of the person
- `Gender` - Gender label
- `Ethnicity` - Ethnicity label
- `Pixels` - Pixel values of the face image

Each face image has a resolution of:

```text
48 × 48 pixels
```

Therefore, each image contains:

```text
48 × 48 = 2,304 pixels
```

For this project:

```text
Input (X)  = Face image pixels
Target (y) = Age
```

Gender and Ethnicity are not used as prediction targets.

---

## Problem Type

This project is a **Regression** problem.

The model predicts a numerical value, which is the person's age.

Example:

```text
Face Image
     ↓
Neural Network
     ↓
Predicted Age = 27.4
```

The goal is to make the predicted age as close as possible to the actual age.

---

## Data Preprocessing

Before training the Neural Network, the dataset is prepared using the following steps.

### 1. Load Pixel Data

The `Pixels` column contains the face image as pixel values.

The pixel data is converted from text into numerical arrays.

### 2. Normalize Pixel Values

Pixel values are normally in the range:

```text
0 - 255
```

The values are normalized to a suitable range for Neural Network training.

### 3. Separate Input and Target

The dataset is separated into:

```text
X = Face image pixels
y = Age
```

---

## Train, Validation and Test

The dataset is divided into three parts:

```text
Training Data
Validation Data
Testing Data
```

### Training Data

Used to train the Neural Network and learn patterns from the face images.

### Validation Data

Used to monitor the model during training and compare different model configurations.

### Testing Data

Used to evaluate the final model with data that was not used during training.

The processed data is saved as `.npy` files in the `outputs` folder.

---

## Neural Network Model

The Neural Network is created in:

```text
nn_model.py
```

The general structure is:

```text
Face Image Pixels
       ↓
Input Layer
       ↓
Hidden Layers
       ↓
Output Layer
       ↓
Predicted Age
```

Because this is a regression problem, the output layer produces one numerical value representing the predicted age.

---

## Neural Network Configuration

The project can test different Neural Network configurations, such as:

- Number of hidden layers
- Number of neurons
- Activation function
- Optimizer
- Learning rate
- Number of epochs

Example:

```text
Model 1
Dense(64)
↓
Output

Model 2
Dense(128)
↓
Dense(64)
↓
Output
```

The purpose is to compare different configurations and find a suitable model.

---

## Epoch Experiment

The project can compare different numbers of epochs.

Example:

```text
10 Epochs
50 Epochs
100 Epochs
```

The model is trained multiple times and the results are compared.

Too few epochs may cause:

```text
Underfitting
```

Too many epochs may cause:

```text
Overfitting
```

The experiment results are saved in:

```text
outputs/experiment_results.csv
```

---

## Training

The training process is:

```text
Training Data
      ↓
Neural Network
      ↓
Prediction
      ↓
Calculate Loss
      ↓
Update Weights
      ↓
Next Epoch
```

The model learns by reducing the difference between the predicted age and the actual age.

---

## Evaluation

Because this is a regression problem, the project evaluates the model using:

- MAE
- MSE
- RMSE
- R² Score

### MAE

**Mean Absolute Error**

Measures the average difference between the actual age and predicted age.

Lower MAE is better.

### MSE

**Mean Squared Error**

Measures the average squared prediction error.

Lower MSE is better.

### RMSE

**Root Mean Squared Error**

Measures prediction error in the same unit as age.

Lower RMSE is better.

### R² Score

Measures how well the model explains the variation in age.

A higher R² generally indicates better performance.

---

## Prediction

After training, the model predicts the age of the test images.

The process is:

```text
Test Image
    ↓
Trained Neural Network
    ↓
Predicted Age
```

The results contain:

```text
Actual Age
Predicted Age
```

For example:

```text
Actual Age    = 25
Predicted Age = 27.4
```

The prediction results are saved in:

```text
outputs/predictions.csv
```

---




## References

### Dataset

Age, Gender and Ethnicity Face Data

https://www.kaggle.com/datasets/nipunarora8/age-gender-and-ethnicity-face-data-csv
