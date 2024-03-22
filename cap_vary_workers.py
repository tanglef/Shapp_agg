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
    ("Shapley", "peerannot"),
    # ("GLAD", "peerannot"),
    ("TwoThird", "peerannot"),
    # ("WDS", "peerannot"),
    (ZeroBasedSkill, "crowdkit"),
    (MMSR, "crowdkit"),
]


datasets = [
    ("bluebirds", 2, 39, 1080),
    ("weathersentiment", 5, 110, 300),
    ("labelme", 8, 77, 1000),
    ("music", 10, 44, 700),
    # ("audiobirds", 2, 205, 79592),
    # ("nist-trec-relevant2", 4, 766, 20232),
    # ("adult2", 5, 825, 11040),
    # ("relevance2", 2, 7138, 99319),
    # ("relevance5", 5, 1273, 363814),
    # ("cifar10h", 10, 2571, 10000),
]

# %%


def refit_json(data, workers, gt):
    data = json_to_dataframe(data)
    data = data[data["worker"].isin(workers.astype(str))]
    tasks = list(data["task"].unique())
    gt = gt[np.array(tasks).astype(int)]
    dic_task = {}
    dic_worker = {}
    refit = {}
    for _, (task, worker, label) in data.iterrows():
        if task not in dic_task:
            dic_task[task] = len(dic_task)
            refit[dic_task[task]] = {}
        if worker not in dic_worker:
            dic_worker[worker] = len(dic_worker)
        refit[dic_task[task]][dic_worker[worker]] = label
    return refit, len(dic_worker), len(dic_task), gt


# %%

results_crowdkit = {
    "strategy": [],
    "n_worker": [],
    "accuracy": [],
    "dataset": [],
    "f1score": [],
}

maxiter = 20
folder = Path.cwd() / "tmp"
folder.mkdir(parents=True, exist_ok=True)
np.random.seed(0)

for data, n_class, n_workers, n_task in tqdm(datasets, desc="Datasets"):
    workers = np.random.choice(range(0, n_workers), 5, replace=False)
    print(f"Processing {data}: {n_class} classes, {n_workers} workers, {n_task} tasks")
    for n_worker in range(5, n_workers // 2, 2):
        while len(workers) < n_worker:
            new_w = np.random.choice(range(0, n_workers), 1, replace=False)
            if new_w not in workers:
                workers = np.append(workers, new_w)
        with open(f"./data/votes_{data}.json") as f:
            votes = json.load(f)
        gt = np.load(f"./data/ground_truth_{data}.npy").astype(int)
        votes, n_worker, n_task, gt = refit_json(votes, workers, gt)
        df = json_to_dataframe(votes)
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
                results_crowdkit["n_worker"].append(n_worker)

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
                results_crowdkit["n_worker"].append(np.nan)

        results_crowdkit_pd = pd.DataFrame(results_crowdkit)
        results_crowdkit_pd.to_csv("result_varying_workers.csv")
        print(results_crowdkit_pd.groupby(by=["dataset", "strategy", "n_worker"]).max())


# %%
rr = results_crowdkit_pd[results_crowdkit_pd["strategy"] != "TwoThird"]
for data, n_class, n_workers, n_task in datasets:
    fig, ax = plt.subplots(1, 1)
    sns.lineplot(
        rr[rr["dataset"] == data],
        x="n_worker",
        y="accuracy",
        hue="strategy",
        ax=ax,
    )
    # plt.yscale("log")
    plt.title(data)

# %%
