import numpy as np
from sys import argv

_, folder = argv

for color in "blue", "yellow", "red":
    for category in "act", "obs", "X":
        result = np.concatenate([np.load("{folder}/block_{color}{conds}_{category}.npy".format(folder=folder, color=color, conds=conds, category=category))
                                    for conds in ("01", "234")])
        np.save("{folder}/result_{color}_{category}.npy".format(folder=folder, color=color, category=category), result)
