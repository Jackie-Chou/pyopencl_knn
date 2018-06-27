import pyopencl as cl
import pyopencl.array
import numpy as np
import argparse

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
parser.add_argument("-k", "--k", type=int, default=1, help="top-k nearest neighbors required")
parser.add_argument("-g", "--gpu", type=int, required=True, help="gpu id to run on")

def parse_input(input_file):
    with np.load(input_file) as data:
        data_mat = data["data_mat"]
    
    return data_mat

def unroll(data_mat):
    """
        unroll the matrix into long vector to pass to opencl kernel
    """
    n, d = data_mat.shape
    data_vec = np.ndarray(shape=(n*d,), dtype=np.float32)
    for i in range(n):
        data_vec[i*d: (i+1)*d] = data_mat[i]

    return data_vec

def roll(vec, output_shape):
    """
        convert the long distance/indice vector into distance/indice matrix of shape (n, n)/k
    """
    n, t = output_shape
    mat = np.ndarray(shape=(n, t), dtype=vec.dtype)
    for i in range(n):
        mat[i] = vec[i*t: (i+1)*t]

    return mat

def main():
    args = parser.parse_args()
    platform = cl.get_platforms()[0]
    print platform

    device = platform.get_devices()[args.gpu]
    print device

    context = cl.Context([device])
    print context

    program = cl.Program(context, open("kernels.cl").read()).build()
    print program

    queue = cl.CommandQueue(context)
    print queue

    data_mat = parse_input(args.input_file)
    n, d = data_mat.shape
    k = args.k
    print("data_mat shape: {}".format(data_mat.shape))
    data_vec = unroll(data_mat)

    mem_flags = cl.mem_flags
    data_buf = cl.Buffer(context, mem_flags.READ_ONLY | mem_flags.COPY_HOST_PTR, \
                           hostbuf=data_vec)
    dist_vec = np.ndarray(shape=(n*n,), dtype=np.float32)
    destination_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, dist_vec.nbytes)

    indice_vec = np.ndarray(shape=(n*k,), dtype=np.int32);
    indice_buf = cl.Buffer(context, mem_flags.WRITE_ONLY, indice_vec.nbytes)

    program.compute_dist(queue, (n, n), None, data_buf, destination_buf, np.int32(d))
    program.find_knn(queue, (n,), None, destination_buf, indice_buf, np.int32(k))

    cl.enqueue_copy(queue, dist_vec, destination_buf)
    cl.enqueue_copy(queue, indice_vec, indice_buf)

    dist_mat = roll(dist_vec, (n, n))
    indice_mat = roll(indice_vec, (n, k))
    print("dist_mat shape: {}".format(dist_mat.shape))
    print("indice_mat shape: {}".format(indice_mat.shape))

    # store to output file
    np.savez(args.output_file, dist_mat=dist_mat, indice_mat=indice_mat)

if __name__ == "__main__":
    main()
