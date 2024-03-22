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

np.random.seed(1)
datasets = [
    ("bluebirds", 2, 39, 1080),
    # ("weathersentiment", 5, 110, 300),
    # ("labelme", 8, 77, 1000),
    # ("music", 10, 44, 700),
    # ("audiobirds", 2, 205, 79592),
    # ("nist-trec-relevant2", 4, 766, 20232),
    # ("adult2", 5, 825, 11040),
    #   ("relevance2", 2, 7138, 99319),
    # ("relevance5", 5, 1273, 363814),
    # ("cifar10h", 10, 2571, 10000),
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
    # ("MV", "peerannot"),
    # ("Wawa", "peerannot"),
    # (DawidSkene, "crowdkit"),
    # (GLAD, "crowdkit"),
    # (KOS, "crowdkit"),
    # (MACE, "crowdkit"),
    ("Shapley", "peerannot"),
    # ("GLAD", "peerannot"),
    # ("TwoThird", "peerannot"),
    # ("WDS", "peerannot"),
    # (ZeroBasedSkill, "crowdkit"),
    # (MMSR, "crowdkit"),
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

        # results_crowdkit_pd = pd.DataFrame(results_crowdkit)
        # results_crowdkit_pd.to_csv("results_crowdkit_shap_once.csv")
        # print(results_crowdkit_pd.groupby(by=["dataset", "strategy"]).max())

# %%
import shap

mod = model.best_model
explainer = shap.TreeExplainer(mod)
shap_values = explainer(model.X_train_np)
shap.plots.beeswarm(shap_values)

# %%
fig = shap.summary_plot(shap_values, model.X_train_np, plot_type="bar", show=False)
plt.savefig("summary_plot_shap_bluebirds.pdf")

# %%
fig = shap.summary_plot(shap_values, model.X_train_np, show=False)
plt.savefig("summary_plot_shap_beeswarm_bluebirds.pdf")


# %%
acc = []
for j in range(39):
    votes_j = np.array([vv[str(j)] for vv in votes.values()])
    acc.append(np.mean(votes_j == gt))
acc
# %%
data_mat = [np.repeat(model.weight.reshape(-1, 1), 2, axis=1) for _ in range(7)]
data_mat = np.array(data_mat)
names = {
    0: "MV",
    1: "WAWA",
    2: "KOS",
    3: "WDS",
    4: "ZeroBasedSkill",
    5: "MMSR",
    6: "Shapley",
}
# Create subplots for heatmaps
fig, axs = plt.subplots(2, 4, figsize=(14, 4))
from matplotlib.colors import LogNorm

# Plot each heatmap
for i in range(4):
    im = axs[0, i].imshow(data_mat[i], cmap="plasma", interpolation="nearest")
    axs[0, i].set_title(f"Strategy {names[i]}")
    plt.colorbar(im, ax=axs[0, i])
    axs[0, i].axis("off")
for i in range(3):
    im = axs[1, i].imshow(
        data_mat[4 + i],
        cmap="plasma",
        interpolation="nearest",
    )
    axs[1, i].set_title(f"Strategy {names[4+i]}")
    if i == 2:
        im = axs[1, i].imshow(
            data_mat[4 + i], cmap="plasma", interpolation="nearest", norm=LogNorm()
        )
        # plt.xscale('log')
        # plt.yscale("log")
    plt.colorbar(im, ax=axs[1, i])
    axs[1, i].axis("off")
# Hide any unused subplots
for i in range(len(data), len(axs)):
    axs[i].axis("off")
axs[1, -1].axis("off")
axs[1, -1].grid(False)

plt.tight_layout()
plt.savefig("matrix_weights_shap_bluebirds.pdf")
plt.show()

# %%
