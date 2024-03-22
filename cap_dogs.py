# %%
import numpy as np
import peerannot
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import crowdkit
from crowdkit.datasets import load_dataset, get_datasets_list
from tqdm.auto import tqdm

sns.set(style="whitegrid")
from crowdkit.aggregation import DawidSkene, GLAD, MACE, MMSR, KOS, ZeroBasedSkill
from sklearn.metrics import accuracy_score, f1_score
from peerannot.models import agg_strategies

# %%

datasets = [
    # ("bluebirds", 2, 39, 1080),
    # ("weathersentiment", 5, 110, 300),
    # ("labelme", 8, 77, 1000),
    # ("music", 10, 44, 700),
    # ("audiobirds", 2, 205, 79592),
    # ("nist-trec-relevant2", 4, 766, 20232),
    # ("adult2", 5, 825, 11040),
    #   ("relevance2", 2, 7138, 99319),
    # ("relevance5", 5, 1273, 363814),
    # ("cifar10h", 10, 2571, 10000),
    ("dog", 4, 109, 807),
]
# %%


def json_to_dataframe(json_data):
    rows = []
    for task, worker_data in json_data.items():
        for worker, label in worker_data.items():
            rows.append({"task": task, "worker": worker, "label": label})
    df = pd.DataFrame(rows)
    return df


strategies = [
    ("MV", "peerannot"),
    ("Wawa", "peerannot"),
    # (DawidSkene, "crowdkit"),
    # (GLAD, "crowdkit"),
    (KOS, "crowdkit"),
    # (MACE, "crowdkit"),
    # ("GLAD", "peerannot"),
    # ("TwoThird", "peerannot"),
    ("WDS", "peerannot"),
    (ZeroBasedSkill, "crowdkit"),
    (MMSR, "crowdkit"),
    ("Shapley", "peerannot"),
]


# %%
results_crowdkit = {"strategy": [], "accuracy": [], "dataset": [], "f1score": []}

maxiter = 20
folder = Path.cwd() / "tmp"
folder.mkdir(parents=True, exist_ok=True)

for data, n_class, n_worker, n_task in tqdm(datasets, desc="Datasets"):
    print(f"Processing {data}: {n_class} classes, {n_worker} workers, {n_task} tasks")
    with open(f"./data/votes_{data}.json") as f:
        votes = json.load(f)
        df = json_to_dataframe(votes)
    gt = np.load(f"./data/ground_truth_{data}.npy").astype(int)
    mask_valid = np.where(gt != -1, True, False)
    for i, (strat, library) in tqdm(enumerate(strategies)):
        try:
            if library == "crowdkit":
                model = strat(n_iter=maxiter)
            else:
                model = agg_strategies[strat]
                model = model(
                    votes,
                    n_workers=n_worker,
                    n_task=n_task,
                    n_classes=n_class,
                    dataset=folder,
                )
            if library == "crowdkit":
                yhat = model.fit_predict(df).to_numpy(dtype=int)
            else:  # peerannot
                if strat in ["Shapley", "GLAD"]:
                    model.run(maxiter=maxiter)
                else:
                    if hasattr(model, "run"):
                        model.run()
                yhat = model.get_answers()
            acc = accuracy_score(gt[mask_valid], yhat[mask_valid])
            f1 = f1_score(gt[mask_valid], yhat[mask_valid], average="macro")
            if library == "crowdkit":
                results_crowdkit["strategy"].append(strat.__name__)
            else:
                results_crowdkit["strategy"].append(strat)
            results_crowdkit["accuracy"].append(acc)
            results_crowdkit["dataset"].append(data)
            results_crowdkit["f1score"].append(f1)
        except Exception as e:
            print(f"Exiting {strat} for {data}")
            print(e)
            if library == "crowdkit":
                results_crowdkit["strategy"].append(strat.__name__)
            else:
                results_crowdkit["strategy"].append(strat)
            results_crowdkit["accuracy"].append(np.nan)
            results_crowdkit["dataset"].append(data)
            results_crowdkit["f1score"].append(np.nan)

        results_crowdkit_pd = pd.DataFrame(results_crowdkit)
        results_crowdkit_pd.to_csv("results_crowdkit_shap_dog.csv")
        print(results_crowdkit_pd.groupby(by=["dataset", "strategy"]).max())
# %%

# data2 = pd.read_csv("results_crowdkit_shap_once.csv")

# for df in data2["dataset"].unique():
#     print()
#     print(df)
#     print(
#         data2[data2["dataset"] == df]
#         .groupby(by=["strategy"])
#         .max()[["accuracy", "f1score"]]
#     )

# %%
