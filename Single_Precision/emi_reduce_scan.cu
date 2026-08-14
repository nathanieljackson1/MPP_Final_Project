#include <stdio.h>
#include <cuda_runtime.h>
#include "support.h"
#include "emi_reduce_scan.h"

#define RS_BLOCK_SIZE 512

// global minimum and maximum reduction kernels
__global__ void reduceMinKernel(const float *in, float *out, unsigned size)
{
    __shared__ float sdata[2 * RS_BLOCK_SIZE];

    unsigned idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    float v0 = (idx < size) ? in[idx] : 1.0e30f;
    float v1 = (idx + blockDim.x < size) ? in[idx + blockDim.x] : 1.0e30f;

    sdata[threadIdx.x] = v0;
    sdata[threadIdx.x + blockDim.x] = v1;

    // Tree reduction to minimum
    for(unsigned stride = 1; stride < (RS_BLOCK_SIZE << 1); stride <<= 1) {
        __syncthreads();
        unsigned index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index < (RS_BLOCK_SIZE << 1)) {
            float a = sdata[index - stride];
            float b = sdata[index];
            sdata[index] = (a < b) ? a : b;
        }
    }

    __syncthreads();
    if(threadIdx.x == 0) {
        out[blockIdx.x] = sdata[(RS_BLOCK_SIZE << 1) - 1];
    }
}

__global__ void reduceMaxKernel(const float *in, float *out, unsigned size)
{
    __shared__ float sdata[2 * RS_BLOCK_SIZE];

    unsigned idx = 2 * blockIdx.x * blockDim.x + threadIdx.x;

    float v0 = (idx < size) ? in[idx] : -1.0e30f;
    float v1 = (idx + blockDim.x < size) ? in[idx + blockDim.x] : -1.0e30f;

    sdata[threadIdx.x] = v0;
    sdata[threadIdx.x + blockDim.x] = v1;

    // Tree reduction to maximum
    for(unsigned stride = 1; stride < (RS_BLOCK_SIZE << 1); stride <<= 1) {
        __syncthreads();
        unsigned index = (threadIdx.x + 1) * stride * 2 - 1;
        if(index < (RS_BLOCK_SIZE << 1)) {
            float a = sdata[index - stride];
            float b = sdata[index];
            sdata[index] = (a > b) ? a : b;
        }
    }

    __syncthreads();
    if(threadIdx.x == 0) {
        out[blockIdx.x] = sdata[(RS_BLOCK_SIZE << 1) - 1];
    }
}

static void launchReduce(const float *d_in, unsigned size, bool isMin, float *h_result)
{
    const float INIT_VAL = isMin ? 1.0e30f : -1.0e30f;

    float *d_src = nullptr;
    float *d_dst = nullptr;
    cudaError_t cuda_ret;

    cuda_ret = cudaMalloc((void**)&d_src, size * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate temp device memory");
    cuda_ret = cudaMemcpy(d_src, d_in, size * sizeof(float), cudaMemcpyDeviceToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy to temp device memory");

    unsigned currSize = size;
    while(currSize > 1) {
        unsigned numBlocks = currSize / (RS_BLOCK_SIZE * 2);
        if(currSize % (RS_BLOCK_SIZE * 2)) numBlocks++;

        cuda_ret = cudaMalloc((void**)&d_dst, numBlocks * sizeof(float));
        if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory for reduction");

        dim3 dim_block(RS_BLOCK_SIZE, 1, 1);
        dim3 dim_grid(numBlocks, 1, 1);

        if(isMin) {
            reduceMinKernel<<<dim_grid, dim_block>>>(d_src, d_dst, currSize);
        } else {
            reduceMaxKernel<<<dim_grid, dim_block>>>(d_src, d_dst, currSize);
        }

        cuda_ret = cudaDeviceSynchronize();
        if(cuda_ret != cudaSuccess) FATAL("Reduction kernel failed");

        cudaFree(d_src);
        d_src = d_dst;
        d_dst = nullptr;
        currSize = numBlocks;
    }

    float hostVal = INIT_VAL;
    cuda_ret = cudaMemcpy(&hostVal, d_src, sizeof(float), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy reduction result to host");

    *h_result = hostVal;

    cudaFree(d_src);
}

void reduceMin(float *d_in, unsigned size, float *h_result)
{
    launchReduce(d_in, size, true, h_result);
}

void reduceMax(float *d_in, unsigned size, float *h_result)
{
    launchReduce(d_in, size, false, h_result);
}

