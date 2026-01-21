
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
    "A40": "prov/BENCH.A.A40",
    "L40S": "prov/BENCH.A.L40S",
    "RTX8000": "prov/BENCH.A.RTX8000",
    "V100": "prov/BENCH.A.V100",
}

all_samples = []

for label, BASE_PATH in BASE_PATHS.items():
    USECASES = [file for file in os.listdir(BASE_PATH) if "compute" in file]

    for USECASE in USECASES:
        PATH = f"{BASE_PATH}/{USECASE}/"
        prov_json = [PATH + file for file in os.listdir(PATH) if file.endswith(".json")]
        if not prov_json:
            continue

        data = json.load(open(prov_json[0], "r"))
        exp_name = prov_json[0].split("/")[-1].removesuffix(".json").removeprefix("prov_")

        ai = get_param(data, exp_name, "yProv4ML:arithmetic_intensity")["$"]
        perf = get_param(data, exp_name, "yProv4ML:performance")["$"]
        memory = get_param(data, exp_name, "yProv4ML:memory")["$"]

        all_samples.append({
            "AI": ai,
            "Perf": perf,
            "memory": memory,
            "GPU": label,
        })

samples = pd.DataFrame(all_samples)
# samples = samples.groupby(["GPU", "memory"]).mean()
# print(samples)

plt.figure(figsize=(8, 6))
plt.title("Roofline Data Points (All Configurations)")

sns.scatterplot(
    data=samples,
    x="AI",
    y="Perf",
    hue="GPU",
    style="GPU",
    markers=True,
    s=80
)

plt.xlabel("Operational Intensity (FLOPs/byte)")
plt.ylabel("Attainable Performance (FLOP/s)")
# plt.xscale("log")
plt.yscale("log")
plt.legend(title="GPU", loc="lower right")

os.makedirs("imgs/combined", exist_ok=True)
plt.savefig("imgs/combined/compute_all_configs.pdf")
plt.show()
