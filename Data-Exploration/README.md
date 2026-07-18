# Dataset exploration 
Dataset exploration is the fist step for Data analyst to understand the structure and quality of the data before performing machine learning or data analysis.  

## Load Dataset
Use `pandas.read_csv()` or `read_excel()` to load the data into a dataframe , where we can analyze and verify the data. 
* install libary `pandas -m pip install pandas` on terminal
```base
import pandas as pd 

df = pd.read_csv("data.csv")	
```

## Display Shape
Use `.shape` to check the number of rows and columns ,verify the shape of the data and determine whether the data is sufficient for analysis.
```base
print(df.shape) 	
```

## Display Data Types
Use `.dtypes` to check the data type of each column ,such as int,float,object or date time 
```base
print(df.dtypes) 
```

## Display Summary Statistics
Use `.describe()` to check summary statistics, such as the mean, median, minimum, maximum, and standard deviation. 
```base
print(df.describe()) 
```

## Display Class Distribution
Use `.value_counts()` on the target column such as `df[‘class’].value_counts()` to verify the dataset has class balance or imbalance  
```base
print(df['class'].value_counts()) 
```
