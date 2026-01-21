import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# SAMPLES
# USECASES = [f for f in os.listdir("prov") if "ml_samples" in f]
# SAMPLES = [u.split("_")[-2] for u in USECASES]
# METRICS = ["Loss"]

# for METRIC in METRICS: 

#     sns.set_style("darkgrid")

#     for USECASE in USECASES: 
#         PATH = f"prov/{USECASE}/metrics_GR0/{METRIC}_Context.TRAINING_GR0.csv"
#         data = pd.read_csv(PATH, sep=",")
#         time = data["None"] - data["None"].min()
#         sns.lineplot(data, y="Context.TRAINING", x=time, alpha=0.5, errorbar=None)

#     # plt.xlim((0, 1000))
#     plt.legend(SAMPLES, title="Number of Samples")
#     plt.savefig(f"imgs/{USECASE}_{METRIC}.pdf")
#     plt.close()


# COMPUTE
USECASES = [f for f in os.listdir("prov") if "ml_" in f]
EPOCHS = [u.split("_")[-2] for u in USECASES]
METRICS = ["Loss"]

for METRIC in METRICS: 

    sns.set_style("darkgrid")

    for USECASE in USECASES: 
        PATH = f"prov/{USECASE}/metrics_GR0/{METRIC}_Context.TRAINING_GR0.csv"
        data = pd.read_csv(PATH, sep=",")
        time = data["None"] - data["None"].min()
        sns.lineplot(data, y="Context.TRAINING", x=time, alpha=0.5, errorbar=None)

    # plt.xlim((0, 1000))
    plt.legend(EPOCHS, title="Epochs")
    plt.ylim(0, 10)
    plt.savefig(f"imgs/{USECASE}_{METRIC}.pdf")
    plt.close()

# PARAMS
# USECASES = [f for f in os.listdir("prov") if "ml_params" in f]
# PARAMS = [u.split("_")[-2] for u in USECASES]
# METRICS = ["Loss"]
# for METRIC in METRICS: 

#     sns.set_style("darkgrid")

#     for USECASE in USECASES: 
#         PATH = f"prov/{USECASE}/metrics_GR0/{METRIC}_Context.TRAINING_GR0.csv"
#         data = pd.read_csv(PATH, sep=",")
#         time = data["None"] - data["None"].min()
#         sns.lineplot(data["Context.TRAINING"], alpha=0.5, errorbar=None)

#     plt.xlim((0, 1000))
#     plt.legend(PARAMS, title="Model Size")
#     plt.savefig(f"imgs/{USECASE}_{METRIC}.pdf")
#     plt.close()