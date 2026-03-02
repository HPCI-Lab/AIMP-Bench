
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

USE_CASE = "C"
NAME = f"time_{USE_CASE}"
import yprov4dv
yprov4dv.start_run(provenance_directory=NAME, run_name=NAME)

all_samples = []
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

    if USE_CASE =="A" and (end-start) / 1000 > 1700:
        continue
    elif USE_CASE =="C" and (end-start) / 1000 > 7000:
        continue

    #"P1": p1,"P2": p2,"P3": p3,
    d = {"time": (end-start) / 1000, "GPU": GPU, "UC": USE_CASE}

    all_samples.append(d)

samples = pd.DataFrame(all_samples)

plt.title(f"USE CASE {USE_CASE}")
sns.boxplot(samples, x="GPU", y="time", hue="GPU")
plt.savefig(f"{NAME}.png", dpi=300)
plt.close()
