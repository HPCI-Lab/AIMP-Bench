
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
    "A40": "prov/BENCH.C.A40",
    "L40S": "prov/BENCH.C.L40S",
    "RTX8000": "prov/BENCH.C.RTX8000",
    "V100": "prov/BENCH.C.V100",
}

all_samples = []

all_pathss = [[os.path.join(B, f) for f in os.listdir(B)] for B in BASE_PATHS.values()]
all_paths = []
for apss in all_pathss: 
    all_paths.extend(apss)

for BASE_PATH in all_paths:
    # USECASES = [file for file in os.listdir(BASE_PATH) if "ml" in file]

    PATH = os.path.join(BASE_PATH, "metrics_GR0", "Loss_Context.TRAINING_GR0.csv")
    # for USECASE in USECASES:
        # PATH = f"{BASE_PATH}/{USECASE}/"

    try: 
        data = pd.read_csv(PATH)
    except: 
        continue
    exp_name = BASE_PATH.split("/")[-1]
    GPU = BASE_PATH.split("/")[-2].split(".")[-1]
    _, p1, p2, p3, _ = exp_name.split("_")

    # epochs = data.groupby("Loss").mean()["Context.TRAINING"]
    epochs = data["Context.TRAINING"][2:25]

    d = {"Params": p1,"Epochs": p2,"Samples": p3, "GPU": GPU}
    for e, v in epochs.items(): 
        d[str(e)] = v

    all_samples.append(d)

samples = pd.DataFrame(all_samples)
# 938 samples per epoch

samples = samples[samples["Epochs"] == "7"].dropna(axis=1)
# samples = samples[samples["Params"]== "1024"]
samples = samples[samples["Samples"]== "1024"]

samples.drop("GPU", inplace=True, axis=1)
samples.drop("Epochs", inplace=True, axis=1)
samples.drop("Samples", inplace=True, axis=1)
samples["Params"] = samples["Params"].astype(int)
samples = samples.groupby(["Params"]).mean()

sns.lineplot(samples.T)
plt.legend(title="Model Params\nper Layer")
plt.xlabel("Epochs")
# plt.xticks(range(0, len(samples.T), 10), range(0, len(samples.T), 10))
plt.ylabel("Loss (MSE)")
plt.savefig("loss.png", dpi=300)
plt.show()

