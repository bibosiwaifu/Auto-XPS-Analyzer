import numpy as np

# 加载原始 12 个样本
X = np.load('X_train.npy')
Y = np.load('Y_train.npy')

def augment_spectrum(spectrum, n_samples=100):
    augmented_x = []
    for _ in range(n_samples):
        # 1. 加入微小的随机位移 (模拟仪器偏差)
        shift = np.random.randint(-10, 10)
        new_spec = np.roll(spectrum, shift)
        
        # 2. 加入随机噪声 (模拟不同信噪比)
        noise = np.random.normal(0, 0.002, spectrum.shape)
        new_spec = new_spec + noise
        
        # 3. 随机增益 (模拟光强波动)
        gain = np.random.uniform(0.9, 1.1)
        new_spec = new_spec * gain
        
        augmented_x.append(np.clip(new_spec, 0, 1))
    return np.array(augmented_x)

X_aug, Y_aug = [], []
print("正在生成增强数据...")
for i in range(len(X)):
    x_variants = augment_spectrum(X[i], n_samples=100) # 每个样本衍生100个
    X_aug.append(x_variants)
    # 对应的 Y 标签保持不变
    for _ in range(100):
        Y_aug.append(Y[i])

X_aug = np.vstack(X_aug)
Y_aug = np.array(Y_aug)

# 保存增强后的数据
np.save('X_aug.npy', X_aug)
np.save('Y_aug.npy', Y_aug)
print(f"增强完成！新数据集维度: X={X_aug.shape}, Y={Y_aug.shape}")