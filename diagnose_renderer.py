import pandas as pd
from zhisa.rendering.chart_renderer import _df_to_image, _fast_render
from PIL import Image
import numpy as np

# Завантажимо шматочок реальних даних
df = pd.read_parquet("d:/zhisa/data/tsdb/BTC_USDT/5m/data.parquet")
window = df.iloc[5000:5064].copy() # 64 свічки

size = 256 # Для наочності збільшимо розмір

# 1. Повільний рендер (Matplotlib)
img_slow = _df_to_image(window, size=size) # [0, 1] RGB
img_slow_u8 = (img_slow * 255).astype(np.uint8)
Image.fromarray(img_slow_u8).save(r"C:\Users\HP\.gemini\antigravity\brain\d7ac3cfe-d4df-46e3-9f3b-75d85638df46\slow_render.png")

# 2. Швидкий рендер (Numpy)
img_fast = _fast_render(window, size=size) # [0, 1] RGB
img_fast_u8 = (img_fast * 255).astype(np.uint8)
Image.fromarray(img_fast_u8).save(r"C:\Users\HP\.gemini\antigravity\brain\d7ac3cfe-d4df-46e3-9f3b-75d85638df46\fast_render.png")

print("Обидва рендери успішно збережено для порівняння!")
