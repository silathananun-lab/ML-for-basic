import pandas as pd
df = pd.read_csv("train.csv")


# ยังไม่ได้ทดสอบ
#--------correlation Heatmap----------
# df = df.drop_duplicates() 

# ยังไม่ได้ทดสอบ
# --------lncorrect Data Correction----------
# df['Sex'] = df['Sex'].replace({'Mle':'Male','Fmale':'Female'})
# เอาค่ากลางมาใส่ให้คนที่กรอกอายุมากกว่า80ปี
# df.loc[df['Age']>80,'Age']=df['Age'].median()

# --------Data Type Convertion----------
# เปลี่ยน data type ของ Age ให้เป็น int เพราะอาจเป็น str
df['Age'] = df['Age'].astype(int)

