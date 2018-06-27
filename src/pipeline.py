import numpy as np
import os
import sys

if __name__ == "__main__":
    argv = sys.argv
    if not len(argv) == 4:
        raise Exception("argv: n, d, k")
    n, d, k = tuple(map(int, argv[1:]))
    data_mat = np.random.randn(n, d).astype(np.float32)
    np.savez("input.npz", data_mat=data_mat)
    os.system("python knn.py -i input.npz -o output.npz -k {} -g 1".format(k))

