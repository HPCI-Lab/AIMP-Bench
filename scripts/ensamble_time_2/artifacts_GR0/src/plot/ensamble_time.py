
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append("./yProv4ML")
import yprov4ml

yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="ensamble_time", 
    provenance_save_dir="scripts",
    collect_all_processes=False, 
    disable_codecarbon=True, 
)
yprov4ml.log_source_code()

def get_param(data, context, param): 
    if param in data["activity"][f"context:{context}"].keys():
        return data["activity"][f"context:{context}"][param]#["prov-ml:parameter_value"]
    return None

# BASE_PATHS = [f"prov/BENCH.A.V100",f"prov/BENCH.A.A40",f"prov/BENCH.A.L40S", f"prov/BENCH.A.A100"]
BASE_PATHS = [os.path.join("prov", p) for p in os.listdir("prov") if p != ".DS_Store"]

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
    GPU = BASE_PATH.split("/")[1].split(".")[2]

    start = int(data["LoggingItemKind.SYSTEM_METRIC"][0])
    end = int(data["LoggingItemKind.SYSTEM_METRIC"][len(data)-1])
    avg_metric = data["Context.TRAINING"].mean()
    if (end-start) / 1000 > 2500: continue

    d = {"time": (end-start) / 1000, "GPU": GPU, "avg_metric": avg_metric}

    all_samples.append(d)

samples = pd.DataFrame(all_samples)
# samples = samples.groupby(["GPU"]).mean()

sns.histplot(samples, y="time", palette="deep")
plt.ylabel("Time (s)")
plt.savefig(f"ensamble_time.png", dpi=300)
plt.close()

yprov4ml.log_artifact("ensamble_time", "ensamble_time.png")
yprov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=False)