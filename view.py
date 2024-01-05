import numpy as np
from matplotlib import pyplot as plt
import os


def check_class(c):
    dirname = f'./processed_data/train/{c}'
    filenames = os.listdir(dirname)
    cnt = 0
    cs = []
    for fn in filenames:
        path = os.path.join(dirname, fn)
        a = np.load(path)
        s = [np.sum(a[i, ::]) for i in range(10)]
        ss = np.sum([_ != 0 for _ in s]).astype(float)
        if ss == 1:
            cnt += 1
            idx = np.argmax(np.asarray(s).astype(float))
            cs.append(idx)

    print(np.bincount(np.asarray(cs).astype(int)))


for i in range(10):
    check_class(i)
# fn = './processed_data/train/0/12575.npy'
# a = np.load(fn)
# for i in range(10):
#     plt.imshow(a[i,::])
#     plt.show()
