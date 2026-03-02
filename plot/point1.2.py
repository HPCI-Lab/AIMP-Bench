

import pandas as pd
import matplotlib.pyplot as plt
import os

import yprov4dv
METRIC = "gpu_usage"
NAME = f"point1.2_{METRIC}"
yprov4dv.start_run(run_name=NAME, provenance_directory=NAME)

USE_CASE = "C"
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
    PATH3 = os.path.join(BASE_PATH, "metrics_GR0", "gpu_usage_Context.TRAINING_GR0.csv")
    GPU = BASE_PATH.split("/")[1].split(".")[-1]
    try: 
        data = pd.read_csv(PATH)
        data2 = pd.read_csv(PATH2)
        data3 = pd.read_csv(PATH3)
    except: 
        continue
    exp_name = BASE_PATH.split("/")[-1]
    _, p1, p2, p3, _ = exp_name.split("_")

    start = int(data["LoggingItemKind.SYSTEM_METRIC"][0])
    end = int(data["LoggingItemKind.SYSTEM_METRIC"][len(data)-1])

    # if p3 == "1024": continue    

    #"P1": p1,"P2": p2,"P3": p3,
    if GPU != "RTX8000": 
        continue
        # data3["Context.TRAINING"] = data3["Context.TRAINING"] / 100.0
    d = {METRIC: data3["Context.TRAINING"][100:900], "time": list(range(len(data3)))[100:900], "GPU": GPU, "P1": p1, "P2": p2, "P3": p3}

    all_samples.append(d)

df = pd.DataFrame(all_samples)

df3 = df[df["P1"] == "1024"]
df2 = df[df["P1"] == "256"]
df1 = df[df["P1"] == "4096"]


for i, d in df3[:1].iterrows(): 
    plt.plot(d[METRIC], label=f"Model size 1024", color="tab:green")

for i, d in df2[:1].iterrows(): 
    plt.plot(d[METRIC], label=f"Model size 256", color="tab:orange")

for i, d in df1[2:3].iterrows(): 
    plt.plot(d[METRIC], label=f"Model size 4096", color="tab:blue")
plt.legend()
plt.title("Use Case C for RTX8000")
plt.ylabel(METRIC)
plt.xlabel("Time")
plt.savefig(f"{NAME}.png", dpi=300)
plt.show()
