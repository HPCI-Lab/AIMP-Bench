
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
sys.path.append("./yProv4ML")
import yprov4ml

gpu_configs = pd.read_csv("./benchmark_params/gpu_config.csv")

yprov4ml.start_run(
    prov_user_namespace="www.example.org",
    experiment_name="ensamble_partition_on_gpu", 
    provenance_save_dir="scripts",
    collect_all_processes=False, 
    disable_codecarbon=True, 
)
yprov4ml.log_source_code()

def get_param(data, context, param): 
    if param in data["activity"][f"context:{context}"].keys():
        return data["activity"][f"context:{context}"][param]#["prov-ml:parameter_value"]
    return None

BASE_PATHS = [f"prov/BENCH.A.V100",f"prov/BENCH.A.A40",f"prov/BENCH.A.L40S", f"prov/BENCH.A.A100"]

#43, 60

PART1 = 20
PART2 = 36

all_samples = []

all_pathss = [[os.path.join(B, f) for f in os.listdir(B)] for B in BASE_PATHS]
all_paths = []
for apss in all_pathss: 
    all_paths.extend(apss)

for BASE_PATH in all_paths:
    PATH = os.path.join(BASE_PATH, "metrics_GR0", "gpu_memory_usage_Context.TRAINING_GR0.csv")
    GPU = BASE_PATH.split("/")[1].split(".")[-1]
    try: 
        data = pd.read_csv(PATH)
    except: 
        continue
    GPU = BASE_PATH.split("/")[1].split(".")[2]

    start = int(data["LoggingItemKind.SYSTEM_METRIC"][0])
    end = int(data["LoggingItemKind.SYSTEM_METRIC"][len(data)-1])
    max_metric = data["Context.TRAINING"].max()
    avg_memory = data["Context.TRAINING"].mean()
    
    time = (end-start) / 1000
    if time > 2500: continue

    memmax = gpu_configs[gpu_configs["GPU Type"] == GPU]["Max Memory Capacity (GB)"].iloc[0]
    memmax = avg_memory * memmax

    if memmax < PART1: 
        partition = 20
    elif memmax > PART2: 
        partition = 36
    elif PART1 < memmax < PART2: 
        partition = 80

    d = {"time": time, "GPU": GPU, "max_memory": memmax, "avg_memory": avg_memory * memmax, "partition": partition}

    all_samples.append(d)

samples = pd.DataFrame(all_samples)["GPU"].value_counts()

GPUS = [f.split(".")[-1] for f in BASE_PATHS]

sns.barplot(samples, palette="deep")
plt.savefig(f"ensamble_partition_on_gpu.png", dpi=300)
plt.close()
yprov4ml.log_artifact(f"ensamble_partition_on_gpu", f"ensamble_partition_on_gpu.png")

yprov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=False)