import pandas as pd

# 1. โหลดข้อมูลไททานิก
df = pd.read_csv('train.csv')

# ==========================================
# STEP 2: ทำ Data Cleaning (จัดการค่าว่าง)
# ==========================================

# 2.1 อายุ (Age) มีค่าว่าง 177 ช่อง -> แทนที่ด้วยค่ากลาง (Median) ของอายุทั้งหมด
df['Age'] = df['Age'].fillna(df['Age'].median())

# 2.2 ท่าเรือ (Embarked) มีค่าว่าง 2 ช่อง -> แทนที่ด้วยท่าเรือที่คนขึ้นเยอะที่สุด (Mode) คือ 'S'
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])


# ==========================================
# STEP 3: ทำ Feature Engineering & Encoding
# ==========================================

# 3.1 แปลงเพศ (Sex) ด้วยวิธี Mapping -> male เป็น 0, female เป็น 1
df['Sex_encoded'] = df['Sex'].map({'male': 0, 'female': 1})

# 3.2 แปลงท่าเรือ (Embarked) ด้วยวิธี One-Hot Encoding -> แตกออกเป็นคอลัมน์ Embarked_C, Embarked_Q, Embarked_S
df_final = pd.get_dummies(df, columns=['Embarked'], dtype=int)

# 3.3 ลบคอลัมน์ที่เป็นข้อความยาวๆ หรือรหัสที่ระบบคำนวณไม่ได้ออกไป
# (เช่น PassengerId, Name, Ticket, Cabin และ Sex เดิมที่ถูกแปลงแล้ว)
columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex']
df_final = df_final.drop(columns=columns_to_drop)


# ==========================================
# STEP 4: บันทึกผลลัพธ์
# ==========================================
df_final.to_csv('cleaned_train.csv', index=False)
print("🎉 ทำความสะอาดและแปลงข้อมูลไททานิกเรียบร้อยแล้ว!")