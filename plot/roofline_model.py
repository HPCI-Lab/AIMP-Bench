
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

# P_peak = 1e4            # 19.5 TFLOP/s
# B_peak = 1e-1           # 900 GB/s

BASE_PATH = "prov/BENCH.A.A40"
# BASE_PATH = "prov/BENCH.A.L40S"
# BASE_PATH = "prov/BENCH.A.RTX8000"
# BASE_PATH = "prov/BENCH.A.V100"
USECASES = [file for file in os.listdir(BASE_PATH) if "compute" in file]

samples = []#{"AI": [], "Perf": [], "memory": []}
for USECASE in USECASES: 
    PATH = f"{BASE_PATH}/{USECASE}/"
    prov_json = [PATH + file for file in os.listdir(PATH) if file.endswith(".json")]
    if len(prov_json) > 0: 
        prov_json = prov_json[0]  
    else: continue
    data = json.load(open(prov_json, "r"))
    exp_name = prov_json.split("/")[-1].removesuffix(".json").removeprefix("prov_")
    ai = get_param(data, exp_name, "yProv4ML:arithmetic_intensity")['$']
    perf = get_param(data, exp_name, "yProv4ML:performance")["$"]
    memory = get_param(data, exp_name, "yProv4ML:memory")["$"]
    # print(ai, perf)
    ts = dict()
    ts["AI"] = ai
    ts["memory"] = memory
    ts["Perf"] = perf
    samples.append(ts)
samples = pd.DataFrame(samples)
print(samples)
samples = samples.sort_values("memory")

ls = np.array(samples["AI"])
B_estimates = np.array(samples["Perf"]) / np.array(samples["AI"])
B_peak_est = np.percentile(B_estimates, 95)
P_peak_est = max(samples["Perf"])

P_mem = B_peak_est * ls
P_roof = np.minimum(P_peak_est, P_mem)

plt.title("Roofline Model")
sns.scatterplot(samples, x="Perf", y="AI", marker="X", color="black", label="ML Runs")
plt.xlabel("Operational Intensity (FLOPS/byte)")
plt.ylabel("Attainable Performance (FLOP/s)")
plt.hlines(P_peak_est, 0, 1e10, colors="red", label="Computational Peak")
plt.plot(ls, P_roof, label="Theoretical Peak")
# plt.xlim(1, 10**15)
plt.xscale("log")
plt.yscale("log")
plt.legend(loc="lower right")
# plt.xticks(range(len(mss)), mss)
os.makedirs(f"imgs/{BASE_PATH}", exist_ok=True)
plt.savefig(f"imgs/{BASE_PATH}/compute.pdf")
plt.show()