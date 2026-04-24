import pandas as pd
import numpy as np
import os
import glob
from scipy.interpolate import interp1d

# =================配置参数=================
DATA_DIR = 'raw_data'          
LABEL_FILE = 'labels_y.csv'        
C1S_REF = 284.8                
TARGET_BE = np.linspace(0, 1200, 12001) # 统一映射到 0.1 eV 步长
OUTPUT_X = 'X_train.npy'
OUTPUT_Y = 'Y_train.npy'

def preprocess_spectrum(file_path):
    valid_data = []
    # 尝试多种编码，确保在 Mac 上能读通
    for enc in ['utf-8', 'utf-16', 'latin-1', 'gbk']:
        try:
            with open(file_path, 'r', encoding=enc, errors='ignore') as f:
                lines = f.readlines()
                if not lines: continue
                
                temp_points = []
                for line in lines:
                    # 跳过包含 "Element" 的表头行
                    if "Element" in line or ";" in line:
                        continue
                        
                    # 按照空格或 Tab 分割数字
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        try:
                            # 尝试转为浮点数
                            be_val = float(parts[0])
                            int_val = float(parts[1])
                            temp_points.append([be_val, int_val])
                        except ValueError:
                            continue
                
                if len(temp_points) > 10:
                    valid_data = temp_points
                    break # 成功读取到数据，跳出编码尝试
        except:
            continue

    if not valid_data:
        print(f"❌ 警告: 无法从 {file_path} 提取数据。请检查文件是否为空。")
        return None
            
    data = np.array(valid_data)
    be = data[:, 0]
    intensity = data[:, 1]

    # 1. 能量轴校准 (寻找 C 1s 峰)
    c1s_mask = (be >= 282) & (be <= 290)
    if any(c1s_mask):
        observed_peak = be[c1s_mask][np.argmax(intensity[c1s_mask])]
        shift = C1S_REF - observed_peak
    else:
        shift = 0 # 如果没找到 C 1s，不偏移
    
    be_calibrated = be + shift

    # 2. 归一化 (减去底噪并缩放到 0-1)
    # 这里使用简单的 Min-Max 归一化
    min_int = np.min(intensity)
    intensity_pure = intensity - min_int
    max_val = np.max(intensity_pure)
    intensity_norm = intensity_pure / max_val if max_val != 0 else intensity_pure

    # 3. 插值 (确保 BE 是递增顺序，interp1d 的要求)
    idx = np.argsort(be_calibrated)
    be_sorted = be_calibrated[idx]
    intensity_sorted = intensity_norm[idx]
        
    f = interp1d(be_sorted, intensity_sorted, bounds_error=False, fill_value=0)
    return f(TARGET_BE)

def main():
    if not os.path.exists(LABEL_FILE):
        print(f"错误: 目录下找不到 {LABEL_FILE}")
        return

    # 读取 Label
    df_labels = pd.read_csv(LABEL_FILE)
    sample_ids = df_labels['Sample_ID'].tolist()
    # 提取数值标签 (At %)
    Y = df_labels.select_dtypes(include=[np.number]).values
    
    X_list = []
    valid_indices = []

    print("🚀 开始处理光谱数据...")
    for i, s_id in enumerate(sample_ids):
        # 兼容 "I/I general.txt" 或 "I general.txt"
        search_pattern = os.path.join(DATA_DIR, str(s_id), "*general*.txt")
        files = glob.glob(search_pattern)
        
        if files:
            txt_path = files[0]
            processed_x = preprocess_spectrum(txt_path)
            if processed_x is not None:
                X_list.append(processed_x)
                valid_indices.append(i)
                print(f"✅ 成功: {s_id} ({os.path.basename(txt_path)})")
        else:
            print(f"❓ 缺失: 找不到样本 {s_id} 的 txt 文件")

    if not X_list:
        print("🚫 失败: 没有成功处理任何样本。")
        return

    X = np.array(X_list)
    Y_final = Y[valid_indices]

    # 保存为 NumPy 格式
    np.save(OUTPUT_X, X)
    np.save(OUTPUT_Y, Y_final)
    
    print("\n" + "="*20)
    print(f"✨ 预处理大功告成！")
    print(f"特征矩阵 X 维度: {X.shape} (样本数, 能量点)")
    print(f"标签矩阵 Y 维度: {Y_final.shape} (样本数, 元素数)")
    print(f"文件已保存: {OUTPUT_X}, {OUTPUT_Y}")
    print("="*20)

if __name__ == "__main__":
    main()
