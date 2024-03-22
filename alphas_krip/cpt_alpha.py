import json
import numpy as np
import pandas as pd
from pathlib import Path

data_dir = Path() / ".." / "data"

from peerannot.models.identification.krippendorff_alpha import Krippendorff_Alpha
from tqdm.auto import tqdm
from crowdkit.metrics.data._classification import alpha_krippendorff


def json_to_dataframe(json_data):
    rows = []
    for task, worker_data in json_data.items():
        for worker, label in worker_data.items():
            rows.append({"task": task, "worker": worker, "label": label})
    df = pd.DataFrame(rows)
    return df


datasets = ["labelme", "music", "relevance2", "bluebirds", "audiobirds", "cifar10h"]
ll = []
for data in tqdm(datasets):
    print(data)
    with open(data_dir / f"votes_{data}.json") as f:
        answers = json.load(f)
        df = json_to_dataframe(answers)
    # k = Krippendorff_Alpha(answers)
    kcrowd = alpha_krippendorff(df)
    print(kcrowd)
    # k.run(data_dir / "identification")


worker = [1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4]
task = [2, 3, 4, 5, 1, 2, 5, 1, 2, 4, 5, 1, 3, 4, 5]
label = [1, 1, 2, 1, 0, 1, 0, 0, 2, 2, 0, 0, 1, 2, 2]
df = pd.DataFrame({"worker": worker, "task": task, "label": label})
print(alpha_krippendorff(df))


# Define the data
data = {
    "task": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] * 4,
    "worker": [
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "A",
        "B",
        "B",
        "B",
        "B",
        "B",
        "B",
        "B",
        "B",
        "B",
        "B",
        "B",
        "B",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "C",
        "D",
        "D",
        "D",
        "D",
        "D",
        "D",
        "D",
        "D",
        "D",
        "D",
        "D",
        "D",
    ],
    "label": [
        1,
        2,
        3,
        3,
        2,
        1,
        4,
        1,
        2,
        None,
        None,
        None,
        1,
        2,
        3,
        3,
        2,
        2,
        4,
        1,
        2,
        5,
        None,
        3,
        None,
        3,
        3,
        3,
        2,
        3,
        4,
        2,
        2,
        5,
        1,
        None,
        1,
        2,
        3,
        3,
        2,
        4,
        4,
        1,
        2,
        5,
        1,
        None,
    ],
}

# Create a Pandas DataFrame
df = pd.DataFrame(data)
df = df.dropna(axis=0)

# Display the DataFrame
print(df)
