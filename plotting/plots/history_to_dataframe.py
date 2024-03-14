"""Module to convert history JSON data to a DataFrame and save it."""

import pandas as pd
import json
from functools import reduce

date_time = "/2024-03-14/15-32-15"
path_to_outputs = "../../outputs" + date_time + "/results/state/histories/history.json"

# These should be the same as in the config
LDA_ALPHA = 100
OPTIMIZER = "SGD"
LEARNING_RATE = 0.03
BP_EXTENSION = "KFLR"
TASK = "CIFAR10"
MODEL = "ResNet18"


# These should stay constant
BATCH_SIZE = 4
CLIENTS = 100
SAMPLED_CLIENTS_PER_ROUND = 20
EVALUATE_CLIENTS_PER_ROUND = 10


with open(path_to_outputs, encoding="utf-8") as f:  # Specify encoding to fix PLW1514
    data = json.load(f)
df_losses_distributed = pd.DataFrame(
    data["losses_distributed"], columns=["Epoch", "Distributed Loss"]
)
df_losses_centralized = pd.DataFrame(
    data["losses_centralized"], columns=["Epoch", "Centralized Loss"]
)

df_metrics_distributed_fit_train_loss = pd.DataFrame(
    data["metrics_distributed_fit"]["train_loss"],
    columns=["Epoch", "Distributed Train Loss"],
)
df_metrics_distributed_fit_train_accuracy = pd.DataFrame(
    data["metrics_distributed_fit"]["train_accuracy"],
    columns=["Epoch", "Distributed Train Accuracy"],
)

df_metrics_distributed = pd.DataFrame(
    data["metrics_distributed"]["test_accuracy"],
    columns=["Epoch", "Distributed Test Accuracy"],
)
df_metrics_centralized = pd.DataFrame(
    data["metrics_centralized"]["test_accuracy"],
    columns=["Epoch", "Centralized Test Accuracy"],
)

combined_df = reduce(
    lambda left, right: left.merge(
        right, on="Epoch", how="outer"
    ),  # Use .merge method to fix PD015
    [
        df_losses_distributed,
        df_losses_centralized,
        df_metrics_distributed_fit_train_loss,
        df_metrics_distributed_fit_train_accuracy,
        df_metrics_distributed,
        df_metrics_centralized,
    ],
)
# remove last row
combined_df = combined_df[:-1]

combined_df["Optimizer"] = OPTIMIZER
combined_df["Learning Rate"] = LEARNING_RATE
combined_df["Batch Size"] = BATCH_SIZE
combined_df["Clients"] = CLIENTS
combined_df["Sampled Clients Per Round"] = SAMPLED_CLIENTS_PER_ROUND
combined_df["Evaluate Clients Per Round"] = EVALUATE_CLIENTS_PER_ROUND
combined_df["LDA Alpha"] = LDA_ALPHA
combined_df["BackPack Extension"] = BP_EXTENSION
combined_df["Task"] = TASK
combined_df["Model"] = MODEL
combined_df["Date"] = date_time

combined_df.to_csv(
    "full-dataset.csv",
    index=False,
    mode="a",
)
