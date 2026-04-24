!pip install gradio -q 

import gradio as gr
import numpy as np
import tensorflow as tf
from scipy.interpolate import interp1d

# --- 准备工作 ---
ELEMENTS = ['Al 2p', 'C 1s', 'Ca 2p3', 'Cl 2p', 'Fe 2p3', 'K 2p3', 'N 1s', 'Na 1s', 'O 1s', 'P 2p', 'S 2p', 'Si 2p', 'Zn 2p3']
TARGET_BE = np.linspace(0, 1200, 12001)

# 【核心修复点】：增加 compile=False
my_model = tf.keras.models.load_model('xps_cnn_model_colab.h5', compile=False)

def predict_xps(file_obj):
    try:
        valid_data = []
        with open(file_obj.name, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if "Element" in line or ";" in line: continue
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        valid_data.append([float(parts[0]), float(parts[1])])
                    except: continue
        
        data = np.array(valid_data)
        be, intensity = data[:, 0], data[:, 1]
        
        intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity) + 1e-8)
        idx = np.argsort(be)
        f = interp1d(be[idx], intensity_norm[idx], bounds_error=False, fill_value=0)
        input_data = f(TARGET_BE).reshape(1, 12001, 1)
        
        # 预测
        prediction = my_model.predict(input_data)[0]
        
        res_dict = {}
        for name, val in zip(ELEMENTS, prediction):
            res_dict[name] = float(max(0, val))
            
        return res_dict
    except Exception as e:
        return {"错误": str(e)}

demo = gr.Interface(
    fn=predict_xps,
    inputs=gr.File(label="上传 XPS 原始 txt 文件 (general.txt)"),
    outputs=gr.Label(num_top_classes=13, label="预测元素含量 (At %)"),
    title="🚀 Auto-XPS-Analyzer (1D-CNN)",
    description="上传你的 Survey Scan 原始文本文件，AI 将自动分析其中的元素组成及含量。"
)

demo.launch(share=True, debug=True)