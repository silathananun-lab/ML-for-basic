# Feature  Engineering 
Feature Engineering is the process of creating, selecting, and transforming features from raw data into a suitable format that enables Machine Learning models to perform computations and generate accurate predictions effectively. 
## Label encording
Label Encoding is a technique that transforms text categories into numeric labels. 

Ex sex column 

male-> 0 

female-> 1 

## One – Hot encording 
Suitable for nominal categorical variables, where categories do not have a natural order or hierarchy, and each category carries equal importance. 

| Embarked | Embarked_C | Embarked_Q | Embarked_S |
| :---: | :---: | :---: | :---: |
| S | 0 | 0 | 1 |
| C | 1 | 0 | 0 |
| S | 0 | 0 | 1 |
| S | 0 | 0 | 1 |
| S | 0 | 0 | 1 |

