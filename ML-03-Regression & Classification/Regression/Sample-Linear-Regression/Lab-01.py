import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error

df = pd.read_csv("ML-03-Regression & Classification/cleaned_train.csv")
data = df[['Fare','Age','Sex_encoded']].dropna()
# Use .dropna() to delete a row have Missing value

# the x value is must 2D array  
x = data[['Fare']] 
# the y value is must 1D array 
y = data['Age']    

# 2. build and train the model 
model = LinearRegression()
model.fit(x, y)
# Use .fit() to Feeds the prepared data into the model for processing and learning relationships within the data
# Instruct the model to learn and find the relationship between X (Fare) and y (Age).
# X (2D array) and y (1D array)

# 3. select a sample data point to compare actual vs predicted values
sample_index = 12
# using iloc[[12]] will return a DataFrame (2D) with the sample data
x_sample = x.iloc[[sample_index]]
y_actual = y.iloc[sample_index]
y_pred = model.predict(x_sample)[0]
# Pass x_sample to the model to predict the age, then use [0].
# Extract only the first predicted value and assign it to a variable.

# Calculate the residual
residual = y_actual - y_pred

# Calculate the mse value for the model
mse_1 = mean_squared_error(y, model.predict(x))

# Display values
print(f"mse_1 = {mse_1}")
print(f"sample index = {sample_index}")
print(f"Actual age = {y_actual}")
print(f"Predict age = {y_pred}")
print(f"Residual = {residual}")

# 4. Plot a graph
fig, ax = plt.subplots(figsize=(8, 6))
# build space for plotting the graph (Figure) and axes (Axes) by setting the width to 8 inches and height to 6 inches

# plot all data points (Observed Data)
ax.scatter(x, y, color='dimgray', alpha=0.7, label='Observed Data', s=25)
# ax.scatter(...): create a scatter plot of all the actual data points in the dataset
# color='dimgray': set the points to a dark gray color
# alpha=0.7: set the transparency of the points to 70%
# s=25: set the size of the points (Size = 25)

# plot Regression Line
X_line = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
y_line = model.predict(X_line)
ax.plot(X_line, y_line, color='#1f77b4', linewidth=2.5, label='Regression Line')
# np.linspace(X.min(), X.max(), 100): build an array of 100 evenly spaced values from the minimum to the maximum Fare
# .reshape(-1, 1): reshape the array to a 2D array to be compatible with the predict() method
# y_line = model.predict(X_line): use the model to predict ages for the range of Fares
# ax.plot(...): draw the regression line in blue (#1f77b4) with a line width of 2.5
    
# plot the actual and predicted points
ax.scatter(x_sample, y_actual, color='red', s=100, zorder=5, label='Actual Value')
ax.scatter(x_sample, y_pred, color='#1f77b4', s=100, zorder=5, label='Predicted Value')
# ax.scatter(x_sample, y_actual, ...): plot the point for the actual value
# ax.scatter(x_sample, y_pred, ...): plot the point for the predicted value
# zorder=5: set the stacking order to ensure the sample point is on top

# draw a red line showing the relationship between the actual and predicted points
ax.plot([x_sample.values[0][0], x_sample.values[0][0]], [y_pred, y_actual], color='red', linestyle='--', linewidth=2)
# ax.plot([x1, x2], [y1, y2], ...): Draw a red dashed line (linestyle='--') conect between the predic point
# with the actuaal point to display the residual

# text Residual
ax.text(x_sample.values[0][0] + 15, (y_actual + y_pred) / 2, f'Residual = {residual:+.1f} years', color='red', fontsize=11, fontweight='bold')
# Write text on the graph at an X-axis position shifted 15 units to the right, and a Y-axis position centered between the actual and predicted values.

# display graph and customize
ax.set_title('Relationship Between Fare and Age', fontsize=14)
ax.set_xlabel('Fare', fontsize=12)
ax.set_ylabel('Age', fontsize=12)
ax.grid(True, linestyle='-', alpha=0.5)
ax.legend(loc='upper left')
plt.tight_layout()
plt.show()
# plt.tight_layout(): Auto-adjust graph layout and margins to prevent text or axis overlap.
# plt.show(): display on the screen