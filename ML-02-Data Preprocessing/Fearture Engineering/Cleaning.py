import pandas as pd

# 1. Load dataset 
df = pd.read_csv("ML-02-Data Preprocessing/train.csv")

# ==========================================
# STEP 2: Data Cleaning 
# ==========================================

# 2.1 Missing Value Handling 
df['Age'] = df['Age'].fillna(df['Age'].median())

df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# ==========================================
# STEP 3: Feature Engineering & Encoding
# ==========================================

# 3.1 Label encording
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})

# 3.2 One-Hot Encoding 
df_final = pd.get_dummies(df, columns=['Embarked'], dtype=int)

# 3.3  PassengerId, Name, Ticket, Cabin and Sex 
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex']
df_final = df_final.drop(columns=columns_to_drop)


# ==========================================
# STEP 4: Save
# ==========================================
df_final.to_csv('cleaned_train.csv', index=False)
print("🎉 Cleaning finish")