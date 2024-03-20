"""Module to convert history JSON data to a DataFrame and save it."""

import pandas as pd
import json
from functools import reduce
import re
from datetime import datetime
from pathlib import Path
import yaml

date_time = "/2024-03-20/15-41-39"
path_to_outputs = "../../outputs" + date_time + "/results/state/histories/history.json"
path_to_yaml = "../../outputs" + date_time + "/.hydra/config.yaml"


def extract_runtime_info(log_content: str, max_rounds: int) -> dict:
    """
    Extract runtime information from log content.

    Args:
        log_content (str): The content of the log file as a string.

    Returns
    -------
        dict: A dictionary containing the runtime information.
    """
    rounds_info: dict = {}

    # Regular expressions to match the relevant log messages
    fit_start_pattern = (
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]\[flwr\]\[DEBUG\] - "
        r"fit_round \d+: strategy sampled \d+ clients \(out of \d+\)"
    )
    fit_start_regex = re.compile(fit_start_pattern)

    fit_end_pattern = (
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]\[flwr\]\[DEBUG\] - "
        r"fit_round \d+ received \d+ results and \d+ failures"
    )
    fit_end_regex = re.compile(fit_end_pattern)

    evaluate_start_pattern = (
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]\[flwr\]\[DEBUG\] - "
        r"evaluate_round \d+: strategy sampled \d+ clients \(out of \d+\)"
    )
    evaluate_start_regex = re.compile(evaluate_start_pattern)

    evaluate_end_pattern = (
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\]\[flwr\]\[DEBUG\] - "
        r"evaluate_round \d+ received \d+ results and \d+ failures"
    )
    evaluate_end_regex = re.compile(evaluate_end_pattern)

    lines = log_content.split("\n")
    for i in range(len(lines)):
        fit_start_match = fit_start_regex.match(lines[i])
        if fit_start_match:
            round_num = int(lines[i].split("fit_round ")[1].split(":")[0].split(" ")[0])
            if 1 <= round_num <= max_rounds:
                fit_start_time = datetime.strptime(
                    fit_start_match.group(1), "%Y-%m-%d %H:%M:%S"
                )
                fit_end_match = fit_end_regex.match(lines[i + 1])
                if fit_end_match:
                    fit_end_time = datetime.strptime(
                        fit_end_match.group(1), "%Y-%m-%d %H:%M:%S"
                    )
                    fit_runtime = (fit_end_time - fit_start_time).total_seconds()
                    rounds_info.setdefault(round_num, {})[
                        "Fit Runtime (s)"
                    ] = fit_runtime

        evaluate_start_match = evaluate_start_regex.match(lines[i])
        if evaluate_start_match:
            round_num = int(lines[i].split("evaluate_round ")[1].split(":")[0])
            if 1 <= round_num <= max_rounds:
                evaluate_start_time = datetime.strptime(
                    evaluate_start_match.group(1), "%Y-%m-%d %H:%M:%S"
                )
                evaluate_end_match = evaluate_end_regex.match(lines[i + 1])
                if evaluate_end_match:
                    evaluate_end_time = datetime.strptime(
                        evaluate_end_match.group(1), "%Y-%m-%d %H:%M:%S"
                    )
                    evaluate_runtime = (
                        evaluate_end_time - evaluate_start_time
                    ).total_seconds()
                    rounds_info.setdefault(round_num, {})[
                        "Evaluate Runtime (s)"
                    ] = evaluate_runtime

    return rounds_info


log_file = "../../outputs" + date_time + "/results/main.log"
log_content = Path(log_file).read_text(encoding="utf-8")

config_info = yaml.safe_load(Path(path_to_yaml).read_text())

LDA_ALPHA = float(config_info["dataset"]["alpha"])
OPTIMIZER = config_info["task"]["fit_config"]["run_config"]["optimizer"]
LEARNING_RATE = float(config_info["task"]["fit_config"]["run_config"]["learning_rate"])
TASK = config_info["task"]["train_structure"]
MODEL = config_info["task"]["model_and_data"].split("_")[1]
BATCH_SIZE = int(config_info["task"]["fit_config"]["dataloader_config"]["batch_size"])
CLIENTS = config_info["fed"]["num_total_clients"]
SAMPLED_CLIENTS_PER_ROUND = config_info["fed"]["num_clients_per_round"]
EVALUATE_CLIENTS_PER_ROUND = config_info["fed"]["num_evaluate_clients_per_round"]
MAX_ROUND_NUM = config_info["fed"]["num_rounds"]

if OPTIMIZER == "diag_exact":
    BP_EXTENSION = "DiagGGNExact"
elif OPTIMIZER == "diag_mc":
    BP_EXTENSION = "DiagGGNMC"
elif OPTIMIZER == "block_exact":
    BP_EXTENSION = "KFLR"
elif OPTIMIZER == "block_mc":
    BP_EXTENSION = "KFAC"
else:
    BP_EXTENSION = "None"

rounds_info = extract_runtime_info(log_content, MAX_ROUND_NUM)
df_rounds_info = pd.DataFrame.from_dict(rounds_info, orient="index")
df_rounds_info["Epoch"] = df_rounds_info.index

with open(path_to_outputs, encoding="utf-8") as f:  # Specify encoding to fix PLW1514
    data = json.load(f)
df_losses_distributed = pd.DataFrame(
    data["losses_distributed"], columns=["Epoch", "Distributed Loss"]
).ffill()
df_losses_centralized = pd.DataFrame(
    data["losses_centralized"], columns=["Epoch", "Centralized Loss"]
).ffill()

df_metrics_distributed_fit_train_loss = pd.DataFrame(
    data["metrics_distributed_fit"]["train_loss"],
    columns=["Epoch", "Distributed Train Loss"],
).ffill()
df_metrics_distributed_fit_train_accuracy = pd.DataFrame(
    data["metrics_distributed_fit"]["train_accuracy"],
    columns=["Epoch", "Distributed Train Accuracy"],
).ffill()

df_metrics_distributed = pd.DataFrame(
    data["metrics_distributed"]["test_accuracy"],
    columns=["Epoch", "Distributed Test Accuracy"],
).ffill()
df_metrics_centralized = pd.DataFrame(
    data["metrics_centralized"]["test_accuracy"],
    columns=["Epoch", "Centralized Test Accuracy"],
).ffill()

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
        df_rounds_info,
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
    f"{TASK}-dataset.csv",
    index=False,
    mode="a",
    header=False,
)
