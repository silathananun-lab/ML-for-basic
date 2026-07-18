# Data VisualiZation 
Data visualization is the process transforming data into visual representation. 
## Histogram
* Use for check the distribution of numerical data.
* This help to show how the data is normal distribution or outlier distribution
* install libary matplotlib `python -m pip install matplotlib` on terminal
```base
import matplotlib.pyplot as plt 

df ['age'].hist(bins=20) 
plt.xlabel("Age") 
plt.ylabel("Frequency") 
plt.show() 	
```
## Correlation Heatmap
* Use for to see the relationship between numerical variables.
* install libart `python -m pip install seaborn` on terminal
```base
import seaborn as sns 
import matplotlib.pyplot as plt 

corr = df.corr(numeric_only=Ture) 
sns.heatmap(corr, annot=True, cmap="coolwarm") 
plt.show() 
```
