import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


USECASES = ["io_large_5", "ml_4096_1_4096_0", "compute_b1024_p1024_1", "ml_1024_1_4096_0"] # "io_15"

d = {}
for USECASE in USECASES: 
    PATH = f"prov/{USECASE}/metrics_GR0/cpu_usage_Context.TRAINING_GR0.csv"
    data = pd.read_csv(PATH, sep=",")

    times = data["LoggingItemKind.SYSTEM_METRIC"]
    duration = max(times) - min(times)
    print(USECASE, duration)
    d[USECASE] = duration

pd.DataFrame(d, index=[0]).plot.bar()
plt.show()


