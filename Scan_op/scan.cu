#include<stdio.h>
#include<stdlib.h>
#include<cuda_runtime.h>

__global__ void global_scan_kernel(float* d_out, float* d_in)
{
    int idx = threadIdx.x;
    d_out[idx] = d_in[idx];
    float out = 0.00f;
    for (int interpre = 1; interpre < sizeof(d_in); interpre *= 2)
    {
        if (idx - interpre >= 0){
            out = d_out[idx] + d_out[idx - interpre];
        }
        __syncthreads();
        if (idx - interpre >= 0){
            d_out[idx] = out;
            out = 0.00f;
        }
    }
}

__global__ void shmem_scan_kernel(float* d_out, float* d_in)
{
    extern __shared__ float d_temp[];
    int idx = threadIdx.x;
    float out = 0.00f;
    d_temp[idx] = d_in[idx];
    __syncthreads();

    for (int i = 1; i < sizeof(d_in); i *= 2){
        if (idx - i >= 0){
            out = d_temp[idx] + d_temp[idx - i];
        }
        __syncthreads();
        if (idx - i >= 0){
            d_temp[idx] = out;
            out = 0.00f;
        }
        __syncthreads();
    }
    d_out[idx] = d_temp[idx];
}


int main(int argc, char** argv)
{
    const int ARRAY_SIZE = 8;
    const int SIZE = ARRAY_SIZE * sizeof(float);

    float h_in[ARRAY_SIZE];
    for (int i = 0; i < ARRAY_SIZE; i++){
        h_in[i] = float(i);
    }
 
    float h_out[ARRAY_SIZE];
    
    float* d_in;
    float* d_out;

    cudaMalloc((void**) &d_in, SIZE);
    cudaMalloc((void**) &d_out, SIZE);

    cudaMemcpy(d_in, h_in, SIZE, cudaMemcpyHostToDevice);

    //int threads = ARRAY_SIZE;
    //int blocks = int(threads / 1024) + 1;
    //printf("Num of blocks: %d\n", blocks);
    //global_scan_kernel<<<blocks, threads>>>(d_out, d_in);
    shmem_scan_kernel<<<1, ARRAY_SIZE, ARRAY_SIZE * sizeof(float)>>>(d_out, d_in);
    
    cudaMemcpy(h_out, d_out, SIZE, cudaMemcpyDeviceToHost);
    
    for (int i = 0; i < ARRAY_SIZE; i++){
        printf("%f", h_out[i]);
        printf(((i%4) != 3) ? "\t" : "\n");
    }

    cudaFree(d_in);
    cudaFree(d_out);

 return 0;
}
