import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# === CONFIG ===
monitor_file = "logs/train/monitor.csv"  # path to your monitor.csv
enable_smoothing = True                  # Set to False to disable smoothing
smooth_window = 10                       # Smoothing window size (only if enabled)

# === LOAD DATA ===
df = pd.read_csv(monitor_file, skiprows=1)  # skip comment line
df['t'] = df['t'] - df['t'].iloc[0]         # normalize time

# === PLOTTING ===
plt.figure(figsize=(10, 5))
plt.plot(df['t'], df['r'], label='Episode Reward', alpha=0.4)

if enable_smoothing:
    smoothed = df['r'].rolling(smooth_window).mean()
    plt.plot(df['t'], smoothed, label=f'{smooth_window}-Episode Moving Average', linewidth=2)

plt.xlabel("Time (s)")
plt.ylabel("Episode Reward")
plt.title("Learning Curve")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
