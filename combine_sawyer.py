import numpy as np
from scipy.misc import imresize

from pickle import dump

def proc_image(img):
    return imresize((img.reshape((480, 480, 3)) + 1) / 2, (64, 64, 3))
def proc_all(x, u, obs):
    assert len(x) == 14 + 6 * 2, str((len(x), 14 + 6 * 2))
    img_flat = proc_image(obs).flatten()
    true_obs = np.concatenate([x[:14], x[14:14 + 3], np.zeros(3*4), x[14 + 6:14 + 9], np.zeros(3*4), img_flat])
    return true_obs, u
def process_file(color, condition):
    stuff = np.load("/home/abhigupta/trajs/{color}_{condition}_0.pkl".format(color=color, condition=condition))
    x, u, obs = stuff['x'], stuff['u'], stuff['obs']
    assert x.shape[:2] == u.shape[:2] == obs.shape[:2]
    result = [[proc_all(x[i][j], u[i][j], obs[i][j]) for j in range(x.shape[1])] for i in range(x.shape[0])]
    with open("/home/abhigupta/trajs/result_{color}_{condition}.pkl".format(color=color, condition=condition), "w") as f:
        dump(result, f)
for color in "red", "blue", "yellow":
    for condition in range(5):
        process_file(color, condition)
