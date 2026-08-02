import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau


# 1. โหลดข้อมูล
print("Loading dataset...")
df = pd.read_csv("ML-03-Regression & Classification/age_gender.csv")
print("Dataset shape:", df.shape)

# 2. แปลง Pixels เป็นรูปภาพ และ Normalize
print("Processing image pixels...")
X = []
for p in df["pixels"]:
    img = np.array(p.split(), dtype=np.float32)
    img = img.reshape(48, 48)
    X.append(img)

X = np.array(X)
X = X / 255.0                  # Normalize ค่าพิกเซลเป็นช่วง 0 - 1
X = X.reshape(-1, 48, 48, 1)   # Reshape สำหรับเข้า CNN (48x48x1)
y = df["age"].values

# แบ่งข้อมูล Train / Test (80% / 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. สร้างโมเดล CNN Regression (ปรับเพิ่มความลึกและป้องกัน Overfitting)
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Fully Connected
    Flatten(),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(1)  # Output node 1 ตัวสำหรับทำนายอายุ
])

# Compile Model
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

model.summary()

# Callback ช่วยปรับ Learning Rate อัตโนมัติเมื่อ Loss นิ่ง
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5, 
    patience=3, 
    min_lr=1e-6,
    verbose=1
)

# 4. เทรนโมเดล
print("\nStarting training...")
history = model.fit(
    X_train,
    y_train,
    epochs=35,              # เพิ่มจำนวน Epochs เพื่อการเรียนรู้ที่ดีขึ้น
    batch_size=64,
    validation_split=0.2,
    callbacks=[reduce_lr]
)

# =========================================================================
# 5. ประเมินผลโมเดล (Evaluation)
# =========================================================================
print("\nEvaluating model on Test Set...")
pred = model.predict(X_test)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))
r2 = r2_score(y_test, pred)

print("\n--- Model Evaluation Results ---")
print(f"MAE  (Mean Absolute Error) : {mae:.4f} ปี")
print(f"RMSE (Root Mean Sq Error)  : {rmse:.4f}")
print(f"R2 Score                  : {r2:.4f}")

# =========================================================================
# 6. ทดสอบทำนายผล + แสดงรูปภาพแบบเนียนคมชัด
# =========================================================================
# เลือกสุ่มรูปจาก Test Set มา 1 รูป
sample_idx = np.random.randint(0, len(X_test))
test_img = X_test[sample_idx]
actual_age = y_test[sample_idx]

# Reshape เตรียมเข้าโมเดล
input_img = test_img.reshape(1, 48, 48, 1)

# ทำนายอายุ
predicted_age = model.predict(input_img)[0][0]

print("\n--- Sample Test Prediction ---")
print(f"Actual Age    : {actual_age} ปี")
print(f"Predicted Age : {predicted_age:.2f} ปี")

# แสดงรูปภาพด้วย Matplotlib พร้อมเกลี่ยพิกเซลเนียน (Bicubic Interpolation)
display_img = test_img.reshape(48, 48)

plt.figure(figsize=(5, 5))
plt.imshow(display_img, cmap='gray', interpolation='bicubic')
plt.title(f"Actual: {actual_age} | Predicted: {predicted_age:.2f}", fontsize=12)
plt.axis('off')
plt.show()