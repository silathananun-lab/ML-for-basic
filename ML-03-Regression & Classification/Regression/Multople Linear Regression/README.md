# Multiple Linear Regression  
Multiple Linear Regression is finding the relationship similar to Simple Linear Regression but different the use a Multiple values to predict result

## Libraries used
```base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error
```

```base
data = df[['Fare','Age','Sex_encoded']].dropna()
```
* Use `.dropna()` to delete a row have Missing value

```base
x = data[['Fare','Sex_encoded']]
y = data['Age']
```
* the x value is must 2D array 
* the y value is must 1D array 

## build and train the model 
```base
model = LinearRegression()
model.fit(x, y)
```
* Use `.fit()` to Feeds the prepared data into the model for processing and learning relationships within the data Instruct the model to learn and find the relationship between X (Fare) and y (Age).

## actual vs predicted values
```base
sample_index = 12
x_sample = x.iloc[[sample_index]]
y_actual = y.iloc[sample_index]
y_pred = model.predict(x_sample)[0]
```
* using `.iloc[[12]]` will return a DataFrame (2D) with the sample data
* Pass x_sample to the model to predict the age, then use [0].
* Extract only the first predicted value and assign it to a variable.

## Calculate the residual
```base
residual = y_actual - y_pred
```

## mse value
```base
mse_2 = mean_squared_error(y, model.predict(x))
```
* mse(mean_squared_error) is A evaluation metric measuring prediction accuracy via the difference between actual and predicted values.

```base
sample_fare = x_sample['Fare'].values[0]
sample_sex = x_sample['Sex_encoded'].values[0]
```
* Use `.values` to Converts a Pandas Series into a NumPy array
* Use [0] to Selects the first element (index 0) from the NumPy array.

```base
fare_range = np.linspace(data['Fare'].min(), data['Fare'].max(), 100)
```
* A series of 100 evenly spaced numbers, ranging from the lowest to the highest Fare values in the dataset.

```base
x_line_df = pd.DataFrame({
    'Fare': fare_range,
    'Sex_encoded': sample_sex
})
y_line = model.predict(x_line_df)
```
* Simulates data to get predicted values (y) for plotting a regression line or decision boundary.