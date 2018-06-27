# pyopencl_knn
a simple k-nearest-neighbour implemented with pyopencl

## file description

check: folder containing opencl code to check the configuration of the machine and opencl

src: folder containing source code

- knn: main code file, configure the host and run opencl. NOTE the platform choice is hard coded to 0, one may feel free to change it. run *python knn.py -h* for arguments and usage.also NOTE this file reads input feature map and kernel from input .npz file, and result will be stored to an output .npz file, example files please refer to *input.npz* and *output.npz*
- kernels.cl: opencl kernel function file
- check.py: a simple correctness checking script by compare the result with a naive serial algorithm. 