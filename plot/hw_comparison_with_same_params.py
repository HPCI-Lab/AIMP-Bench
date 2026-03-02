

import pandas as pd
import matplotlib.pyplot as plt
import os

import yprov4dv
USE_CASE = "C"
NAME = f"hw_comparison_with_same_params_{USE_CASE}"
yprov4dv.start_run(provenance_directory=NAME, run_name=NAME)

def get_param(data, context, param): 
    if param in data["activity"][f"context:{context}"].keys():
        return data["activity"][f"context:{context}"][param]#["prov-ml:parameter_value"]
    return None

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
    if USE_CASE != "C": 
        _, p1, p2, _ = exp_name.split("_")
    else: 
        _, p1, p2, p3, _ = exp_name.split("_")

    start = int(data["LoggingItemKind.SYSTEM_METRIC"][0])
    end = int(data["LoggingItemKind.SYSTEM_METRIC"][len(data)-1])

    if USE_CASE == "A": 
        if p2 == "p4096": continue
    elif USE_CASE == "C": 
        if p3 == "1024": continue

    #"P1": p1,"P2": p2,"P3": p3,
    if USE_CASE != "C": 
        d = {"time": (end-start) / 1000, "GPU": GPU, "P1": p1, "P2": p2}
    else: 
        d = {"time": (end-start) / 1000, "GPU": GPU, "P1": p1, "P2": p2, "P3": p3}
    
    all_samples.append(d)

df = pd.DataFrame(all_samples)

if USE_CASE == "C": 

    df_filtered = (df.groupby(["P1", "P2", "P3"]).filter(lambda x: x["GPU"].nunique() > 1))
    pivot = df_filtered.pivot_table(index=["P1", "P2", "P3"],columns="GPU",values="time")

    p3_values = sorted(pivot.index.get_level_values("P3").unique())
    n_plots = len(p3_values)
    n_cols = 1
    n_rows = n_plots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows), sharey=False)
    if n_plots == 1:
        axes = [axes]
    for ax, p3 in zip(axes, p3_values):
        subset = pivot.xs(p3, level="P3")
        subset.plot(kind="bar", ax=ax)
        ax.set_title(f"Num. Samples = {p3 if p3 != "None" else "All"}")
        ax.set_ylabel("Time (seconds)")
        ax.set_xlabel("(Model Size, Epochs)")
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(f"hw_comparison_with_same_params_{USE_CASE}.png", dpi=300)
else: 

    df_filtered = (df.groupby(["P1", "P2"]).filter(lambda x: x["GPU"].nunique() > 1))
    pivot = df_filtered.pivot_table(index=["P1", "P2"],columns="GPU",values="time")

    pivot.plot(kind="bar", figsize=(12,6))
    plt.ylabel("Time (seconds)")
    plt.title(f"HW Time per (Augmentation Func. Size, Batch Size) in Use Case {USE_CASE}")
    plt.xticks(rotation=45)
    plt.xlabel("(Augmentation Func. Size, Batch Size)")
    plt.tight_layout()
    plt.savefig(f"hw_comparison_with_same_params_{USE_CASE}.png", dpi=300)
