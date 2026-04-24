import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# 1. 加载增强后的数据
X = np.load('X_aug.npy')
Y = np.load('Y_aug.npy')

# CNN 需要 3D 输入: (样本数, 长度, 通道数)
X = X.reshape(X.shape[0], X.shape[1], 1)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2. 构建 1D-CNN 模型 (参考 SOP 架构)
model = models.Sequential([
    layers.Conv1D(32, 7, activation='relu', input_shape=(X.shape[1], 1)),
    layers.MaxPooling1D(4),
    layers.Conv1D(64, 5, activation='relu'),
    layers.MaxPooling1D(4),
    layers.Conv1D(128, 3, activation='relu'),
    layers.GlobalAveragePooling1D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(Y.shape[1], activation='linear') # 预测百分比用 linear
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. 训练
print("开始训练 1D-CNN...")
history = model.fit(X_train, y_train, 
                    epochs=50, 
                    batch_size=32, 
                    validation_data=(X_test, y_test))

# 4. 评估
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\n测试集平均绝对误差 (MAE): {test_mae:.4f}")

model.save('xps_cnn_model.h5')
print("模型已保存为 xps_cnn_model.h5")
