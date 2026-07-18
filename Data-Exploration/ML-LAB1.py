import pandas as pd
df = pd.read_csv("train.csv")

# Display shape
print(df.shape)
# results->(891,12)

# Display Data type
print(df.dtypes)
# results->   PassengerId      int64
#             Survived         int64
#             Pclass           int64
#             Name               str
#             Sex                str
#             Age            float64
#             SibSp            int64
#             Parch            int64
#             Ticket             str
#             Fare           float64
#             Cabin              str
#             Embarked           str
#             dtype: object

# Display summary statistics
print(df.describe)
# results-->
#           PassengerId  Survived  Pclass  ... Fare Cabin  Embarked
# 0              1         0       3  ...   7.2500   NaN         S
# 1              2         1       1  ...  71.2833   C85         C
# 2              3         1       3  ...   7.9250   NaN         S
# 3              4         1       1  ...  53.1000  C123         S
# 4              5         0       3  ...   8.0500   NaN         S
# ..           ...       ...     ...  ...      ...   ...       ...
# 886          887         0       2  ...  13.0000   NaN         S
# 887          888         1       1  ...  30.0000   B42         S
# 888          889         0       3  ...  23.4500   NaN         S
# 889          890         1       1  ...  30.0000  C148         C
# 890          891         0       3  ...   7.7500   NaN         Q

# Display Missing Values
print(df.isnull())
# results-->
    #        PassengerId  Survived  Pclass  Name  Sex    Age  SibSp  Parch  Ticket   Fare   Cabin  Embarked
    # 0          False     False   False  False  False  False  False  False   False  False   True     False
    # 1          False     False   False  False  False  False  False  False   False  False  False     False
    # 2          False     False   False  False  False  False  False  False   False  False   True     False
    # 3          False     False   False  False  False  False  False  False   False  False  False     False
    # 4          False     False   False  False  False  False  False  False   False  False   True     False
    # ..           ...       ...     ...    ...    ...    ...    ...    ...     ...    ...    ...       ...
    # 886        False     False   False  False  False  False  False  False   False  False   True     False
    # 887        False     False   False  False  False  False  False  False   False  False  False     False
    # 888        False     False   False  False  False   True  False  False   False  False   True     False
    # 889        False     False   False  False  False  False  False  False   False  False  False     False
    # 890        False     False   False  False  False  False  False  False   False  False   True     False

print(df.isnull().sum())
# results-->
# PassengerId      0
# Survived         0
# Pclass           0
# Name             0
# Sex              0
# Age            177
# SibSp            0
# Parch            0
# Ticket           0
# Fare             0
# Cabin          687
# Embarked         2
# dtype: int64

# Display Duolicate recouds
print(df.duplicated())
# results-->
# 0      False
# 1      False
# 2      False
# 3      False
# 4      False
#        ...  
# 886    False
# 887    False
# 888    False
# 889    False
# 890    False
# Length: 891, dtype: bool

print(df.duplicated().sum())
# results--> 0

# Display Class Distribution
print(df['Pclass'].value_counts())
# results-->
# Pclass
# 3    491
# 1    216
# 2    184