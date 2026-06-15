import pandas as pd
from zhisa.rendering.chart_renderer import render_chart
from PIL import Image
import numpy as np

# Load a piece of the real BTCUSDT dataset we just downloaded
df = pd.read_parquet("d:/zhisa/data/tsdb/BTC_USDT/5m/data.parquet")

# Take a 128-bar window
window = df.iloc[5000:5128].copy()

# Render it using the model's exact renderer
# Default size for S1 is usually 64, but let's render at 256 so the user can see it clearly
tensor = render_chart(window, size=256)

# Convert tensor (3, H, W) back to numpy image
# tensor is in [0, 1] range
img_array = tensor.permute(1, 2, 0).numpy()
img_array = (img_array * 255).astype(np.uint8)

# Save image
img = Image.fromarray(img_array)
img.save(r"C:\Users\HP\.gemini\antigravity\brain\d7ac3cfe-d4df-46e3-9f3b-75d85638df46\render.png")
print("Render saved successfully!")
