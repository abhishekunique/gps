import numpy as np
from scipy.misc import imresize
from sys import argv

def reshape(image):
    return imresize(image[32:].reshape((480, 480, 3)), (64, 64, 3))[:,:,::-1]

_, folder = argv

for color in "blue", "yellow", "red":
    for category in "act", "obs", "X":
        result = np.concatenate([np.load("{folder}/block_{color}{conds}_{category}.npy".format(folder=folder, color=color, conds=conds, category=category))
                                    for conds in ("01", "234")])
        if category == "obs":
            result = reshaped = np.array([[reshape(img) for img in y] for y in result])
        np.save("{folder}/result_{color}_{category}.npy".format(folder=folder, color=color, category=category), result)
