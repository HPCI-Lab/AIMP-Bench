
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('benchmark_params/gpu_dataset.csv')

print(df["GPU Avg Power Consumption (W)"].mean(), df["GPU Avg Power Consumption (W)"].std())

sns.scatterplot(df, x="GPU Memory Request (Gb)", y="GPU Avg Power Consumption (W)")
plt.show()