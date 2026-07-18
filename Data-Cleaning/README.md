# Data Cleaning 
Data cleaning is the process of identifying and fixing errors in data to ensure it is correct and ready for analysis. 
## Missing Value Handling  
* steps in the data preparation process  to data analysis or training a machine learning model. 
```base
df['Age'] = df['Age'].fillna(['Age'].mean())
```
## Duplicate Removal
* the duplicate data can distort the analyze results.
* Use `.drop_duplicates()` to delete the duplicate rows 
```base
df = df.drop_duplicates() 
```
## Incorrect Data Correction
* Use .replace() to fix the incorrect value.
* Use `.log[]` to modify specific rows.
```base
 	df['gender'] = df['gender'].replace({'Mle':'Male', 	'Fmale':'Female'}) 

df.loc[df['age'] > 120, 'age'] = df['age'].median() 
```
## Data Type Conversion
* some the data in column may contain incorrect.
* `Use .astype()` or `pd.to_datetime()`
```base
df['age'] = df['age'].astype(int) 
df['date'] = pd.to_datetime(df['date']) 
```
### Compare 
* **Mean** -> all value / number of data
* **Median** -> median value 

Comparing the mean with the Median help identify the data contains outliers.     