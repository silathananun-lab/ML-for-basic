import pandas as pd
df = pd.read_csv("train.csv")

#--------Missing Value Handling----------
# add Age with the Mean
df['Age'] = df['Age'].fillna(['Age'].mean())

print("\n--- missing Value data ---")
print(df)

#--------correlation Heatmap----------
df = df.drop_duplicates() 

# --------lncorrect Data Correction----------
df['Sex'] = df['Sex'].replace({'Mle':'Male','Fmale':'Female'})
df.loc[df['Age']>80,'Age']=df['Age'].median()

# --------Data Type Convertion----------
df['Age'] = df['Age'].astype(int)

