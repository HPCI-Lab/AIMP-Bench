
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

import yprov4dv
USE_CASE = "A"
GPU_C = "RTX8000"
METRIC = "gpu_usage"
NAME = f"point2.2_{USE_CASE}_{GPU_C}_{METRIC}"
yprov4dv.start_run(run_name=NAME, provenance_directory=NAME)

BASE_PATHS = { GPU_C: f"prov/BENCH.{USE_CASE}.{GPU_C}"}

all_pathss = [[os.path.join(B, f) for f in os.listdir(B)] for B in BASE_PATHS.values()]
all_paths = []
for apss in all_pathss: 
    all_paths.extend(apss)

all_samples = []
for BASE_PATH in all_paths:
    PATH = os.path.join(BASE_PATH, "metrics_GR0", f"{METRIC}_Context.TRAINING_GR0.csv")
    GPU = BASE_PATH.split("/")[1].split(".")[-1]
    try: 
        data = pd.read_csv(PATH)
    except: 
        continue
    exp_name = BASE_PATH.split("/")[-1]
    _, p1, p2, _ = exp_name.split("_")

    start = int(data["LoggingItemKind.SYSTEM_METRIC"][0])
    end = int(data["LoggingItemKind.SYSTEM_METRIC"][len(data)-1])

    # if p2 == "p4096": continue
    data = data["Context.TRAINING"].mean()
    if GPU in ["A40", "L40S"]: # "L40S"
        data /= 100.0

    d = {METRIC: data ,"time": (end-start) / 1000, "GPU": GPU, "P1": p1[1:], "P2": p2[1:]}

    all_samples.append(d)

df = pd.DataFrame(all_samples)
# df4 = df[df["P1"] == "1024"]
# df3 = df[df["P1"] == "512"]
# df2 = df[df["P1"] == "256"]
# df1 = df[df["P1"] == "128"]
# df0 = df[df["P1"] == "64"]

# VAR = "gpu_usage"
# for i, d in df4.iterrows(): 
#     print(d[VAR].max(), d[VAR].min(), d["P1"], d["P2"])
#     plt.plot(d[VAR], label=f"Batch size 1024", color="tab:green")

# for i, d in df3.iterrows(): 
#     print(d[VAR].max(), d[VAR].min(), d["P1"], d["P2"])
#     if d[VAR].max() == d[VAR].min(): continue
#     plt.plot(d[VAR], label=f"Batch size 512", color="tab:orange")

# for i, d in df2.iterrows(): 
#     print(d[VAR].max(), d[VAR].min(), d["P1"], d["P2"])
#     plt.plot(d[VAR], label=f"Batch size 256", color="black")

# for i, d in df1.iterrows(): 
#     print(d[VAR].max(), d[VAR].min(), d["P1"], d["P2"])
#     plt.plot(d[VAR], label=f"Batch size 128", color="tab:red")

# for i, d in df0.iterrows(): 
#     print(d[VAR].max(), d[VAR].min(), d["P1"],d["P2"])
#     plt.plot(d[VAR], label=f"Batch size 64", color="tab:blue")


sns.scatterplot(df, x="time", y=METRIC, hue="P1", style="P2")

plt.legend()
plt.title(f"Use Case {USE_CASE} for {GPU_C}")
plt.ylabel(METRIC)
plt.xlabel("Time")
plt.tight_layout()
plt.savefig(f"{NAME}.png", dpi=300)
plt.show()
