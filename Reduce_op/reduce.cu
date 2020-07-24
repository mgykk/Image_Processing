#include<stdio.h>
#include<cuda_runtime.h>
#include<stdlib.h>

__global__ void global_reduce_kernel(float* d_out, float* d_in)
{
    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s)
        {
            d_in[myId] += d_in[myId + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        d_out[blockIdx.x] = d_in[myId];
    }

}

__global__ void shmem_reduce_kernel(float* d_out, const float* d_in)
{
    extern __shared__ float sdata[];

    int myId = blockDim.x * blockIdx.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = d_in[myId];
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1){
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0){
        d_out[blockIdx.x] = sdata[0];
    }
}

void reduce(float* d_out, float* d_intermediate, float* d_in, int size, bool usesSharedMemory)
{
    const int maxThreadsPerBlock = 1024;
    int threads = maxThreadsPerBlock;
    int blocks = size / maxThreadsPerBlock;
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_intermediate, d_in);
    }
    else{
        global_reduce_kernel<<<blocks, threads>>>(d_intermediate, d_in);
    }
    threads = blocks;
    blocks = 1;
    if (usesSharedMemory)
    {
        shmem_reduce_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_out, d_intermediate);
    }
    else{
        global_reduce_kernel<<<blocks, threads>>>(d_out, d_intermediate);
    }
}

int main(int argc, char **argv)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0){
        fprintf(stderr, "error:no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }
    int dev = 0;
    cudaSetDevice(dev);
    
    cudaDeviceProp devProps;
    if (cudaGetDeviceProperties(&devProps, dev) == 0)
    {
        printf("Using device %d:\n", dev);
        printf("%s; global mem: %dB; compute v%d.%d; clock: %d kHz\n", devProps.name, (int)devProps.totalGlobalMem, (int)devProps.major, (int)devProps.minor, (int)devProps.clockRate);
    }

    const int ARRAY_SIZE = 1 << 20;
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(float);
 
    float h_in[ARRAY_SIZE];
    float sum = 0.0f;
    for (int i = 0; i < ARRAY_SIZE; i++){
        h_in[i] = -1.0f + (float)random()/((float)RAND_MAX/2.0f);  //[-1,1]
        sum += h_in[i];
    }
    float * d_in, * d_intermediate, * d_out;

    cudaMalloc((void **) &d_in, ARRAY_BYTES);
    cudaMalloc((void **) &d_intermediate, ARRAY_BYTES);
    cudaMalloc((void **) &d_out, sizeof(float));

    cudaMemcpy(d_in, h_in, ARRAY_SIZE, cudaMemcpyHostToDevice);

    int whichKernel = 0;
    if (argc == 2){
        whichKernel = atoi(argv[1]);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    switch(whichKernel){
    case 0:
        printf("Running gloal reduce\n");
        cudaEventRecord(start, 0);
        for(int i = 0; i < 100; i++){
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, false);
        }
        cudaEventRecord(stop, 0);
        break;
    case 1:
        printf("Running reduce with shared mem\n");
        cudaEventRecord(start, 0);
        for (int i = 0; i < 100; i++)
        {
            reduce(d_out, d_intermediate, d_in, ARRAY_SIZE, true);
        }
        cudaEventRecord(stop, 0);
        break;
    default:
        fprintf(stderr, "error: ran no kernel\n");
        exit(EXIT_FAILURE);
    }
    cudaEventSynchronize(stop);
    float elapsedTime;
    cudaEventElapsedTime(&elapsedTime, start, stop);
    elapsedTime /= 100.0f;
    float h_out;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    printf("average time elapsed: %f\n", elapsedTime);

    cudaFree(d_in);
    cudaFree(d_intermediate);
    cudaFree(d_out);
    return 0;
}


























