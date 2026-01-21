import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

USECASES = [f for f in os.listdir("prov/") if "io_" in f]
METRICS = ["cpu_usage", "gpu_usage", "memory_usage"]

for METRIC in METRICS: 
    sns.set_style("darkgrid")

    for USECASE in USECASES: 
        PATH = f"prov/{USECASE}/metrics_GR0/{METRIC}_Context.TRAINING_GR0.csv"

        data = pd.read_csv(PATH, sep=",")

        sns.lineplot(data["Context.TRAINING"], label=USECASE)
    plt.legend()
    plt.savefig(f"imgs/io_bench_{METRIC}.pdf")
    plt.close()
    # plt.show()


d = {}
for USECASE in USECASES: 
    PATH = f"prov/{USECASE}/metrics_GR0/memory_usage_Context.TRAINING_GR0.csv"

    data = pd.read_csv(PATH, sep=",")

    d[USECASE] = max(data["LoggingItemKind.SYSTEM_METRIC"] - min(data["LoggingItemKind.SYSTEM_METRIC"]))

d = pd.DataFrame(d, index=[0])
d.plot.bar()
plt.legend()
plt.savefig(f"imgs/io_bench_time.pdf")
plt.close()
