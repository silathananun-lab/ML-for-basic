import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df = pd.read_csv("ML-03-Regression & Classification/cleaned_train.csv")

data = df[['Fare','Age','Sex_encoded']].dropna()

x = data[['Fare','Sex_encoded']]
y = data['Age']

model = LinearRegression()
model.fit(x,y)

sample_index = 12
x_sample = x.iloc[[sample_index]]
y_actual = y.iloc[sample_index]
y_pred = model.predict(x_sample)[0]

residual = y_actual - y_pred
mse_2 = mean_squared_error(y, model.predict(x))
print(f"mse_2 = {mse_2}")
print(f"sample index = {sample_index}")
print(f"Actual age = {y_actual}")
print(f"Predict age = {y_pred}")
print(f"Residual = {residual}")

sample_fare = x_sample['Fare'].values[0]
sample_sex = x_sample['Sex_encoded'].values[0]

fare_range = np.linspace(data['Fare'].min(), data['Fare'].max(), 100)

x_line_df = pd.DataFrame({
    'Fare': fare_range,
    'Sex_encoded': sample_sex
})
y_line = model.predict(x_line_df)


fig, ax = plt.subplots(figsize=(8, 6))

ax.scatter(data['Fare'], y, color='dimgray', alpha=0.5, label='Observed Data', s=25)
ax.plot(fare_range, y_line, color='#1f77b4', linewidth=2.5, label='Regression Line (for sample sex)')

ax.scatter(sample_fare, y_actual, color='red', s=100, zorder=5, label='Actual Value')
ax.scatter(sample_fare, y_pred, color='#1f77b4', s=100, zorder=5, label='Predicted Value')

# Draw a red dashed line 
ax.plot([sample_fare, sample_fare], [y_pred, y_actual], color='red', linestyle='--', linewidth=2)

# display Residual value
ax.text(sample_fare + 5, (y_actual + y_pred) / 2, 
        f'Residual = {residual:+.1f} years', color='red', fontsize=11, fontweight='bold')

ax.set_title('Relationship Between Fare and Age Sex', fontsize=14)
ax.set_xlabel('Fare', fontsize=12)
ax.set_ylabel('Age', fontsize=12)
ax.grid(True, linestyle='-', alpha=0.5)
ax.legend(loc='upper right')

plt.tight_layout()
plt.show()
