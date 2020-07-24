//导入cuda所需的运行时库

#include "cuda_runtime.h"



#include <stdio.h>

#include<stdlib.h>

#include<cmath>

#include<corecrt_wstdio.h>

#include "device_launch_parameters.h"

#include "malloc.h"





__global__ void vectorADD(const float *A, const float *B, float *C, int numElements)

{

	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < numElements) {

		C[i] = A[i] + B[i];

	}

}



int main(void)

{

	// A/B/C元素总数

	int numElements = 5000;

	size_t size = numElements * sizeof(float);

	printf("Vector addition of %d elements.\n", numElements);



	//在CPU端给ABC三个向量申请存储空间

	float *h_A = (float *)malloc(size);

	float *h_B = (float *)malloc(size);

	float *h_C = (float *)malloc(size);

	//初始化

	for (int i = 0; i < numElements; ++i) {

		h_A[i] = rand() / (float)RAND_MAX;

		h_B[i] = rand() / (float)RAND_MAX;

	}

	//在GPU中给ABC三个向量申请空间

	float *d_A = NULL;

	float *d_B = NULL;

	float *d_C = NULL;

	cudaMalloc((void **)&d_A, size);

	cudaMalloc((void **)&d_B, size);

	cudaMalloc((void **)&d_C, size);

	//把数据AB从CPU内存当中复制到GPU显存中

	printf("Copy input data from the host memory to device memory\n");

	cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

	//执行GPU kernel函数

	int threadsPerBlock = 256;

	int blockPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;

	vectorADD <<<blockPerGrid, threadsPerBlock >>> (d_A, d_B, d_C, numElements);

	cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

	for (int i = 0; i < numElements; ++i) {

		if (fabs(h_A[i] + h_B[i] - h_C[i]) > 1e-5) {

			fprintf(stderr, "result verification falied at element %d!\n", i);

			exit(EXIT_FAILURE);

		}

	}

	cudaFree(d_A);

	cudaFree(d_B);

	cudaFree(d_C);

	free(h_A);

	free(h_B);

	free(h_C);

	printf("test passed\n");

	return 0;

}
