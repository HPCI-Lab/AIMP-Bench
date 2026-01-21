
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

def get_param(data, context, param): 
    if param in data["activity"][f"context:{context}"].keys():
        return data["activity"][f"context:{context}"][param]#["prov-ml:parameter_value"]
    return None

BASE_PATHS = {
    "A40": "prov/BENCH.B.A40",
    "L40S": "prov/BENCH.B.L40S",
    "RTX8000": "prov/BENCH.B.RTX8000",
    "V100": "prov/BENCH.B.V100",
}

METRICS = ["gpu_usage"] # "gpu_usage", "cpu_usage", "memory_usage", "gpu_power_usage"]
RUN_SIZE = "small"
# BS = "128"
GPUC = "V100"

all_samples = []

all_pathss = [[os.path.join(B, f) for f in os.listdir(B)] for B in BASE_PATHS.values()]
all_paths = []
for apss in all_pathss: 
    all_paths.extend(apss)

for BASE_PATH in all_paths:
    exp_name = BASE_PATH.split("/")[-1]
    GPU = BASE_PATH.split("/")[-2].split(".")[-1]
    _, p1, p2, _ = exp_name.split("_")
    if GPU != GPUC: continue
    if p1 != RUN_SIZE: continue
    d = {"size": p1, "batch_size": p2, "GPU": GPU}
    
    for metric in METRICS: 
        try: 
            PATH = os.path.join(BASE_PATH, "metrics_GR0", f"{metric}_Context.TRAINING_GR0.csv")
            data = pd.read_csv(PATH)
        except: 
            continue

        data = data["Context.TRAINING"].dropna()#.mean()
        for i, it in data.items(): 
            d[str(i)] = it

    all_samples.append(d)

samples = pd.DataFrame(all_samples)

# samples = samples.groupby(["GPU"])#[METRICS]#.mean()
samples.drop("size", inplace=True, axis=1)
samples.drop("GPU", inplace=True, axis=1)
# samples.drop("batch_size", inplace=True, axis=1)
samples.index = samples["batch_size"].astype(int)
samples.drop("batch_size", inplace=True, axis=1)
plt.figure(figsize=(4,5))
sns.boxplot(samples.T)
plt.ylabel(metric)
plt.ylim((0, 0.5))
plt.title(f"GPU: {GPUC}; I/O: {RUN_SIZE}")
plt.tight_layout()
plt.savefig(f"{metric}_{RUN_SIZE}_{GPUC}.png", dpi=300)
plt.show()

