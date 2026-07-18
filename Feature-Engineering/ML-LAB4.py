import pandas as pd
df = pd.read_csv('train.csv')

print("--- Unencoded data ---")
print(df[['Name', 'Sex', 'Embarked']].head(5))
print("\n" + "="*50 + "\n")


# ==========================================
# 2. Label Encoding (Sex column)
# ==========================================
# Using Pandas .map() into Text to numbers
#  male = 0 , female = 1
df['Sex_LabelEncoded'] = df['Sex'].map({'male': 0, 'female': 1})

print("--- ผลลัพธ์หลังทำ Label Encoding (คอลัมน์ Sex) ---")
print(df[['Name', 'Sex', 'Sex_LabelEncoded']].head(5))
print("\n" + "="*50 + "\n")


# ==========================================
# 3.One-Hot Encoding (Embarked cloumn)
# ==========================================
# using Missing Value in the Embarked cloumn because the Embarked cloumn have 2 Missing Value
# using Median
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# Using pd.get_dummies() to ecpand the Embarked cloumn
# Using dtype=int to output numericle values 
df_one_hot = pd.get_dummies(df, columns=['Embarked'], dtype=int)

# display encoded data
print("--- encoded data ---")
encoded_columns = ['Name', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
print(df_one_hot[encoded_columns].head(5))


# ==========================================
# 4. save encoded data
# ==========================================
# Combine both into a single DataFrame before saving.
df_one_hot['Sex_LabelEncoded'] = df['Sex_LabelEncoded']
df_one_hot.to_csv('titanic_encoded_example.csv', index=False)
print("\n🎉 save finish")