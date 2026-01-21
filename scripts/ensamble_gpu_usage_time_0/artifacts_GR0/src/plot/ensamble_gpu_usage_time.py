
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append("./yProv4ML")
import yprov4ml

yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="ensamble_gpu_usage_time", 
    provenance_save_dir="scripts",
    collect_all_processes=False, 
    disable_codecarbon=True, 
)
yprov4ml.log_source_code()

def get_param(data, context, param): 
    if param in data["activity"][f"context:{context}"].keys():
        return data["activity"][f"context:{context}"][param]#["prov-ml:parameter_value"]
    return None

BASE_PATHS = [f"prov/BENCH.A.V100",f"prov/BENCH.B.V100",f"prov/BENCH.C.V100"]

all_samples = []

all_pathss = [[os.path.join(B, f) for f in os.listdir(B)] for B in BASE_PATHS]
all_paths = []
for apss in all_pathss: 
    all_paths.extend(apss)

for BASE_PATH in all_paths:
    PATH = os.path.join(BASE_PATH, "metrics_GR0", "gpu_usage_Context.TRAINING_GR0.csv")
    GPU = BASE_PATH.split("/")[1].split(".")[-1]
    try: 
        data = pd.read_csv(PATH)
    except: 
        continue
    exp_name = BASE_PATH.split("/")[1].split(".")[1]
    # _, p1, p2, p3, _ = exp_name.split("_")

    start = int(data["LoggingItemKind.SYSTEM_METRIC"][0])
    end = int(data["LoggingItemKind.SYSTEM_METRIC"][len(data)-1])
    avg_metric = data["Context.TRAINING"].mean()

    #"P1": p1,"P2": p2,"P3": p3,
    d = {"time": (end-start) / 1000, "USE": exp_name, "avg_metric": avg_metric}

    all_samples.append(d)

samples = pd.DataFrame(all_samples)
# samples = samples.groupby(["GPU"]).mean()

sns.jointplot(samples, x="avg_metric", y="time", palette="deep", hue="USE")
plt.xlabel("GPU Usage (%)")
plt.ylabel("Time (s)")
plt.savefig(f"scatter.png", dpi=300)
plt.close()

yprov4ml.log_artifact("scatter", "scatter.png")
yprov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=False)