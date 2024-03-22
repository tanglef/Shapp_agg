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
    ("bluebirds", 2, 39, 1080),
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

n_worker = 39
K = 2

import matplotlib.pyplot as plt
import numpy as np

# Generate sample data for heatmaps
data_mat = [np.random.rand(n_worker, K) for _ in range(7)]
data_mat[0] = 0 * data_mat[0] + 1  # MV

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
        if library == "crowdkit":
            strat = strat.__name__
        if strat == "Wawa":
            data_mat[1] = np.repeat(model.worker_score.reshape(-1, 1), 2, axis=1)
        elif strat == "KOS":
            # need to assign self.kos_data = kos_data in crowd-kit source code
            model.kos_data["abs_relab"] = np.abs(model.kos_data["reliabilities"])
            data_mat[2] = np.repeat(
                model.kos_data.groupby("worker")
                .mean()["abs_relab"]
                .values.reshape(-1, 1),
                2,
                axis=1,
            )
        elif strat == "WDS":
            for j in range(n_worker):
                data_mat[3][j] = np.array([model.pi[j][0, 0], model.pi[j][1, 1]])
        elif strat == "ZeroBasedSkill":
            model.skills_.index = model.skills_.index.astype(int)
            data_mat[4] = np.repeat(
                model.skills_.sort_index().values.reshape(-1, 1), 2, axis=1
            )
        elif strat == "MMSR":
            model.skills_.index = model.skills_.index.astype(int)
            data_mat[5] = np.repeat(
                model.skills_.sort_index().values.reshape(-1, 1), 2, axis=1
            )
        elif strat == "Shapley":
            data_mat[6] = np.repeat(model.weight.reshape(-1, 1), 2, axis=1)
for i in range(len(data_mat)):
    data_mat[i] = data_mat[i].T
# %%
# data_mat = np.array(data_mat)
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
# plt.savefig("matrix_weights_bluebirds.pdf")
plt.show()

# %%
