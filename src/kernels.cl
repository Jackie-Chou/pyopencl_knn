__kernel void compute_dist(__global const float *data_vec, 
                           __global float *dist_vec,
                           const int dim)
{
    int n, i, j;
    n = get_global_size(0);
    i = get_global_id(0);
    j = get_global_id(1);
    
    float dist = 0.0f;
    int k;
    float tem;

    for(k = 0; k < dim; ++ k)
    {
        tem = data_vec[i*dim+k] - data_vec[j*dim+k];
        dist += tem * tem;
    }

    dist_vec[i*n + j] = sqrt(dist);
}

__kernel void find_knn(__global const float *dist_vec,
                       __global int *indice_vec,
                       const int k)
{
    int n, i, j;
    n = get_global_size(0);
    i = get_global_id(0);

    // virtual bubble sort
    int a, b;
    int cnt;
    int indice;

    cnt = 0;
    for(a = i*n; a < i*n+k; ++ a)
    {
        indice = (i+1)*n-1;
        for(b = (i+1)*n-1; b > a; -- b)
        {
            // virtual swap, only record the indice
            if(dist_vec[indice] > dist_vec[b-1])
                indice = b-1;
        }
        indice_vec[i*k+cnt] = indice - i*n;
        cnt ++;
    }
}


