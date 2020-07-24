#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void histo_kernel(int* d_out, int* d_in, int out_size)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int id_temp = d_in[idx];
    int my_idx = id_temp % out_size;
    atomicAdd(&(d_out[my_idx]), 1);
}

int main(int argc, char** argv)
{
    int ARRAY_SIZE = 4096;  //65536
    int out_size = 16;
    int SIZE = ARRAY_SIZE * sizeof(int);

    int h_in[ARRAY_SIZE];
    int h_out[out_size];

    for (int i = 0; i < ARRAY_SIZE; i++){
        h_in[i] = i;  //bit_reverse(i, log2(ARRAY_SIZE))
    }

    for (int i = 0; i < out_size; i++){
        h_out[i] = 0;  //bit_reverse(i, log2(ARRAY_SIZE))
    }

    int* d_in;
    int* d_out;

    cudaMalloc((void**) &d_in, SIZE);
    cudaMalloc((void**) &d_out, out_size * sizeof(int));

    cudaMemcpy(d_in, h_in, SIZE, cudaMemcpyHostToDevice);

    int threads = 1024;
    int blocks;
    if (ARRAY_SIZE % threads == 0){
        blocks = int(ARRAY_SIZE / threads);
    }else{
        blocks = int(ARRAY_SIZE / threads) + 1;
    }

    printf("threads num: %d; blocks num: %d\n",threads, blocks);

    histo_kernel<<<blocks, threads>>>(d_out, d_in, out_size);

    cudaMemcpy(h_out, d_out, out_size * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < out_size; i++){
        printf("Count %d: %d\n", i, h_out[i]);
    }
    
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
