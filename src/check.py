"""
    check the correctness of knn
"""
import numpy as np
import argparse
import os

class customFormatter(argparse.RawTextHelpFormatter, argparse.ArgumentDefaultsHelpFormatter):
    pass
parser = argparse.ArgumentParser(formatter_class=customFormatter)
parser.add_argument("-i", "--input_file", type=str, required=True, help="input numpy file (.npz) containing the data points matrix"
                                                                       "\nNOTE the data points matrix should be a numpy ndarray of shape (n, d)"
                                                                       "\nand dtype np.float32, its name should be 'data_mat'")
parser.add_argument("-o", "--output_file", type=str, required=True, help="output numpy file (.npz) containing the output distance matrix and"
                                                                        "\ntop-k nearest neighbor indice matrix"
                                                                        "\nNOTE the distance matrix is a ndarray of shape (n, n)"
                                                                        "\nand dtype np.float32, its name is 'dist_mat'"
                                                                        "\ndist_mat[i,j] represents the distance between i-th and j-th data vector"
                                                                        "\nthe indice matrix is a ndarray of shape (n, k)"
                                                                        "\nand dtype np.int32, its name is 'indice_mat'"
                                                                        "\nindice_mat[i,j] represents indice of the j-th nearest neighbor of data i")

if __name__ == "__main__":
    args = parser.parse_args()
    with np.load(args.input_file) as data:
        data_mat = data['data_mat']
    with np.load(args.output_file) as data:
        dist_mat = data['dist_mat']
        indice_mat = data['indice_mat']

    n, d = data_mat.shape
    k = indice_mat.shape[1]
    # compute dist_mat2
    dist_mat2 = np.ndarray(shape=(n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            dist_mat2[i, j] = np.linalg.norm(data_mat[i] - data_mat[j])

    # compute indice_mat2
    indice_mat2 = np.ndarray(shape=(n, k), dtype=np.int32)
    for i in range(n):
        for a in range(k):
            indice = n - 1
            for b in range(a, n):
                if dist_mat2[i, b] < dist_mat2[i, indice]:
                    indice = b
            indice_mat2[i, a] = indice

    print("average squared error of dist_mat: {}".format(np.linalg.norm(dist_mat - dist_mat2)))
    print("indice all same: {}".format(np.array_equal(indice_mat, indice_mat2)))



