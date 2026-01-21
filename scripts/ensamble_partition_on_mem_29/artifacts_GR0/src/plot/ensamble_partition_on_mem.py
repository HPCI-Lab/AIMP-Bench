
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
    experiment_name="ensamble_partition_on_mem", 
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

#43, 60

PART1 = 20
PART2 = 30
PART3 = 40

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
        partition = PART1
    elif PART1 < memmax < PART2: 
        partition = PART2
    elif PART2 < memmax < PART3: 
        partition = PART3
    elif memmax > PART3: 
        partition = 60
    else: 
        print("problem")

    d = {"time": time, "GPU": GPU, "max_memory": memmax, "avg_memory": avg_memory * memmax, "partition": partition}

    all_samples.append(d)

samples = pd.DataFrame(all_samples)["partition"].value_counts()

GPUS = [f.split(".")[-1] for f in BASE_PATHS]

# for gpu in GPUS: 
# sample = samples[samples["GPU"] == gpu]
sns.barplot(samples, palette="deep")
plt.ylabel("GPU Memory Requested (Gb)")
plt.savefig(f"ensamble_partition_on_mem.png", dpi=300)
plt.close()
yprov4ml.log_artifact(f"ensamble_partition_on_mem", f"ensamble_partition_on_mem.png")

yprov4ml.end_run(create_graph=True, create_svg=True, crate_ro_crate=False)