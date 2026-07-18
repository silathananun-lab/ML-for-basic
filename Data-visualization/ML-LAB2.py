import pandas as pd
# libary histrogram
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("train.csv")

# --------Histogram----------
# df['Age'].hist(bins=20,rwidth=0.85)
# สั่งเจาะจงไปที่คอลัมน์อายุ ['Age'] ในตาราง แล้วสร้างกราฟฮิสโตแกรม (.hist) 
# โดยแบ่งช่วงข้อมูลออกเป็น 20 แท่ง (bins=20) เพื่อดูว่าคนช่วงอายุไหนมีจำนวนเยอะหรือน้อย
# rwidth=0.85 ระยะห่างระหว่างเเท่ง
# plt.xlabel("Age")
# ใส่ชื่อกำกับที่ แกนนอน (แกน X) ของกราฟ โดยให้แสดงคำว่า "Age"
# plt.ylabel("Frequency")
# ใส่ชื่อกำกับที่ แกนนอน (แกน y ของกราฟ โดยให้แสดงคำว่า "Frequency"
# plt.show()
# สั่งให้ระบบ เปิดหน้าต่าง Pop-up แสดงรูปกราฟ ขึ้นมาบนหน้าจอคอมพิวเตอร์ของคุณ 
# ถ้าไม่มีบรรทัดนี้ กราฟจะถูกวาดเสร็จแค่ในความทรงจำของคอมพิวเตอร์ แต่จะไม่ยอมแสดงออกมาให้เราเห็นครับ

# -------correlation Heatmap----------
# corr =df.corr(numeric_only="True")
# numeric_only="True" เฉพาะข้อมูลตัวเลข
# sns.heatmap(corr,annot=True,cmap="coolwarm")
# plt.show()

