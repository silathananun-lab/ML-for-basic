import pandas as pd

# ==========================================
# 1. โหลดข้อมูลจากไฟล์ train.csv
# ==========================================
df = pd.read_csv('train.csv')

print("--- ข้อมูลเดิมก่อนทำ Encoding ---")
print(df[['Name', 'Sex', 'Embarked']].head(5))
print("\n" + "="*50 + "\n")


# ==========================================
# 2. ตัวอย่างการทำ Label Encoding (คอลัมน์ Sex)
# ==========================================
# ใช้ฟังก์ชัน .map() ของ Pandas ในการกำหนดตัวเลขให้ข้อความโดยตรง
# ให้ male = 0 และ female = 1
df['Sex_LabelEncoded'] = df['Sex'].map({'male': 0, 'female': 1})

print("--- ผลลัพธ์หลังทำ Label Encoding (คอลัมน์ Sex) ---")
print(df[['Name', 'Sex', 'Sex_LabelEncoded']].head(5))
print("\n" + "="*50 + "\n")


# ==========================================
# 3. ตัวอย่างการทำ One-Hot Encoding (คอลัมน์ Embarked)
# ==========================================
# ก่อนทำต้องจัดการค่าว่าง (Missing Value) ในคอลัมน์ Embarked ก่อน (มีว่างอยู่ 2 ช่อง)
# โดยเติมด้วยค่าฐานนิยม 'S' (ท่าเรือที่คนขึ้นเยอะที่สุด)
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])

# ใช้ฟังก์ชัน pd.get_dummies() เพื่อแตกคอลัมน์ Embarked ออกเป็นคอลัมน์เฉพาะของแต่ละท่าเรือ
# ใส่ dtype=int เพื่อให้แสดงผลเป็นตัวเลข 1 และ 0 (ถ้าไม่ใส่จะเป็น True/False)
df_one_hot = pd.get_dummies(df, columns=['Embarked'], dtype=int)

# แสดงเฉพาะคอลัมน์ที่เกี่ยวข้องมาดูผลลัพธ์
print("--- ผลลัพธ์หลังทำ One-Hot Encoding (คอลัมน์ Embarked) ---")
encoded_columns = ['Name', 'Embarked_C', 'Embarked_Q', 'Embarked_S']
print(df_one_hot[encoded_columns].head(5))


# ==========================================
# 4. บันทึกข้อมูลที่แปลงเสร็จแล้วลงไฟล์ใหม่
# ==========================================
# รวมทั้งสองอย่างไว้ใน DataFrame เดียวกันก่อนบันทึก
df_one_hot['Sex_LabelEncoded'] = df['Sex_LabelEncoded']
df_one_hot.to_csv('titanic_encoded_example.csv', index=False)
print("\n🎉 บันทึกผลลัพธ์ลงไฟล์ 'titanic_encoded_example.csv' เรียบร้อยแล้ว!")