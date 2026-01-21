
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

USE_CASE = "A"
BASE_PATHS = { 
    "A40": f"prov/BENCH.{USE_CASE}.A40",
    "L40S": f"prov/BENCH.{USE_CASE}.L40S",
    "RTX8000": f"prov/BENCH.{USE_CASE}.RTX8000",
    "V100": f"prov/BENCH.{USE_CASE}.V100",
}

all_samples = []

all_pathss = [[os.path.join(B, f) for f in os.listdir(B)] for B in BASE_PATHS.values()]
all_paths = []
for apss in all_pathss: 
    all_paths.extend(apss)

for BASE_PATH in all_paths:
    PATH = os.path.join(BASE_PATH, "metrics_GR0", "cpu_usage_Context.TRAINING_GR0.csv")
    GPU = BASE_PATH.split("/")[1].split(".")[-1]
    try: 
        data = pd.read_csv(PATH)
    except: 
        continue
    exp_name = BASE_PATH.split("/")[-1]
    # _, p1, p2, p3, _ = exp_name.split("_")

    start = int(data["LoggingItemKind.SYSTEM_METRIC"][0])
    end = int(data["LoggingItemKind.SYSTEM_METRIC"][len(data)-1])

    #"P1": p1,"P2": p2,"P3": p3,
    d = {"time": (end-start) / 1000, "GPU": GPU}

    all_samples.append(d)

samples = pd.DataFrame(all_samples)
samples = samples.groupby(["GPU"]).mean()

sns.barplot(samples, x="GPU", y="time", palette="deep")
# plt.show()
plt.title(f"Use Case {USE_CASE}")
plt.savefig(f"time_{USE_CASE}.pdf")
plt.close()
