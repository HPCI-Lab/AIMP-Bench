

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import yprov4dv
USE_CASE = "B"
NAME = f"point2.1_{USE_CASE}"
yprov4dv.start_run(run_name=NAME, provenance_directory=NAME)

BASE_PATHS = { 
    "A40": f"prov/BENCH.{USE_CASE}.A40",
    "L40S": f"prov/BENCH.{USE_CASE}.L40S",
    "RTX8000": f"prov/BENCH.{USE_CASE}.RTX8000",
    "V100": f"prov/BENCH.{USE_CASE}.V100",
}

all_pathss = [[os.path.join(B, f) for f in os.listdir(B)] for B in BASE_PATHS.values()]
all_paths = []
for apss in all_pathss: 
    all_paths.extend(apss)

all_samples = []
for BASE_PATH in all_paths:
    PATH = os.path.join(BASE_PATH, "metrics_GR0", "gpu_memory_usage_Context.TRAINING_GR0.csv")
    PATH2 = os.path.join(BASE_PATH, "metrics_GR0", "memory_usage_Context.TRAINING_GR0.csv")
    PATH3 = os.path.join(BASE_PATH, "metrics_GR0", "cpu_usage_Context.TRAINING_GR0.csv")
    GPU = BASE_PATH.split("/")[1].split(".")[-1]
    try: 
        data = pd.read_csv(PATH)
        data2 = pd.read_csv(PATH2)
        data3 = pd.read_csv(PATH3)
    except: 
        continue
    exp_name = BASE_PATH.split("/")[-1]
    _, p1, p2, _ = exp_name.split("_")

    start = int(data["LoggingItemKind.SYSTEM_METRIC"][0])
    end = int(data["LoggingItemKind.SYSTEM_METRIC"][len(data)-1])
    data = data["Context.TRAINING"].mean()
    data2 = data2["Context.TRAINING"].mean()
    data3 = data3["Context.TRAINING"].mean()

    if USE_CASE == "A":
        if p2 == "p4096": continue

    d = {"time": (end-start) / 1000, "GPU": GPU, "P1": p1, "P2": p2}

    all_samples.append(d)

df = pd.DataFrame(all_samples)
if USE_CASE == "A": 
    df["batch_size"] = df["P1"].str.extract(r'(\d+)').astype(int)
else: 
    df["batch_size"] = df["P2"].astype(int)
df = df.sort_values("batch_size")

plt.figure(figsize=(10,6))

sns.lineplot(
    data=df,
    x="batch_size",
    y="time",
    hue="GPU",
    marker="o"
)

plt.xlabel("Batch Size (log scale)")
plt.ylabel("Time")
plt.title(f"Use Case {USE_CASE} Time vs Batch Size (Separated by GPU)")
plt.xscale("log")
plt.tight_layout()
plt.savefig(f"{NAME}.png", dpi=300)
plt.show()