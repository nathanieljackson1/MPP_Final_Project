#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "support.h"
#include "kernel.cu"
#include "emi_reduce_scan.h"

int main(int argc, char* argv[])
{
    // Problem Setup ////////////////////
    Timer timer;

    printf("\nSetting up the problem..."); fflush(stdout);
    startTime(&timer);

    double *L_h, *C_h, *R_h, *fsw_h;
    double *worstAtten_h, *thd_h;
    double *L_d, *C_d, *R_d, *fsw_d;
    double *worstAtten_d, *thd_d;

    unsigned numL, numC, numR, numFsw, totalCombinations;
    unsigned numHarmonics;
    unsigned blockSize, coarseFactor;
    int version;
    cudaError_t cuda_ret;
    dim3 dim_grid, dim_block;

    if(argc == 1) {
        numL = 50; 
        numC = 50; 
        numR = 10; 
        numFsw = 10;
        version = 1;
        blockSize = 256;
        coarseFactor = 4;
    } else if(argc == 8) {
        numL = atoi(argv[1]);
        numC = atoi(argv[2]);
        numR = atoi(argv[3]);
        numFsw = atoi(argv[4]);
        version = atoi(argv[5]);
        blockSize = atoi(argv[6]);
        coarseFactor = atoi(argv[7]);
    } else {
        printf("\n    Invalid input parameters!"
        "\n");
        exit(0);
    }

    if(blockSize == 0 || blockSize > 1024 || (blockSize % 32) != 0) {
        FATAL("Block size must be a positive multiple of 32, up to 1024");
    }
    if(coarseFactor == 0) {
        FATAL("Coarse factor must be at least 1");
    }
    if(version == 5 && (numR > MAX_CONST_PARAM || numFsw > MAX_CONST_PARAM)) {
        FATAL("numR and numFsw must each be <= 2048 for the constantmemory version (Kernel 5)");
    }

    numHarmonics = 15;
    totalCombinations = numL * numC * numR * numFsw;

    initParamVector(&L_h, numL, 1e-6, 1e-3);
    initParamVector(&C_h, numC, 1e-9, 1e-5);
    initParamVector(&R_h, numR, 0.01, 10.0);
    initParamVector(&fsw_h, numFsw, 1e4, 2e5);

    worstAtten_h = (double*)malloc(totalCombinations * sizeof(double));
    thd_h = (double*)malloc(totalCombinations * sizeof(double));
    if(worstAtten_h == NULL || thd_h == NULL) FATAL("Unable to allocate host");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    Sweep size = %u (L:%u x C:%u x R:%u x Fsw:%u), Version = %d, BlockSize = %u, CoarseFactor = %u\n", totalCombinations, numL, numC, numR, numFsw, version, blockSize, coarseFactor);

    // Variable Allocation on Device ////////////////////
    printf("Allocating device variables..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMalloc((void**)&L_d, numL * sizeof(double));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&C_d, numC * sizeof(double));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&R_d, numR * sizeof(double));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&fsw_d, numFsw * sizeof(double));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&worstAtten_d, totalCombinations * sizeof(double));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&thd_d, totalCombinations * sizeof(double));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // H2D Data Transfer ////////////////////
    printf("Copying data from host to device..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(L_d, L_h, numL * sizeof(double), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(C_d, C_h, numC * sizeof(double), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(R_d, R_h, numR * sizeof(double), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");
    cuda_ret = cudaMemcpy(fsw_d, fsw_h, numFsw * sizeof(double), cudaMemcpyHostToDevice);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to the device");

    cuda_ret = cudaMemset(worstAtten_d, 0, totalCombinations * sizeof(double));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");
    cuda_ret = cudaMemset(thd_d, 0, totalCombinations * sizeof(double));
    if(cuda_ret != cudaSuccess) FATAL("Unable to set device memory");

    if(version == 5) {
        cuda_ret = cudaMemcpyToSymbol(R_const, R_h, numR * sizeof(double));
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy R to constant memory");
        cuda_ret = cudaMemcpyToSymbol(fsw_const, fsw_h, numFsw * sizeof(double));
        if(cuda_ret != cudaSuccess) FATAL("Unable to copy Fsw to constant memory");
    }

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // Run Selected Kernel ////////////////////
    printf("Launching kernel (version %d)...", version); fflush(stdout);
    startTime(&timer);

    dim_block.x = blockSize; dim_block.y = dim_block.z = 1;

    unsigned LCper = numL * numC;
    unsigned effectiveCoarseFactor = (version == 4) ? coarseFactor : 1;
    unsigned blockSpan = effectiveCoarseFactor * blockSize;
    unsigned maxGroupSpan = (blockSpan / LCper) + 2;
    size_t sharedMemBytes = (size_t)(2 * maxGroupSpan) * sizeof(double);

    if(version == 1) {
        dim_grid.x = (totalCombinations + blockSize - 1) / blockSize;
        dim_grid.y = dim_grid.z = 1;
        evaluateFilterKernel_naive<<<dim_grid, dim_block>>>(L_d, C_d, R_d, fsw_d, numL, numC, numR, numFsw, numHarmonics, worstAtten_d, thd_d);
    } else if(version == 2) {
        dim_grid.x = (totalCombinations + blockSize - 1) / blockSize;
        dim_grid.y = dim_grid.z = 1;
        evaluateFilterKernel_shared<<<dim_grid, dim_block, sharedMemBytes>>>(L_d, C_d, R_d, fsw_d, numL, numC, numR, numFsw, numHarmonics, worstAtten_d, thd_d);
    } else if(version == 3) {
        dim_grid.x = (totalCombinations + blockSize - 1) / blockSize;
        dim_grid.y = dim_grid.z = 1;
        evaluateFilterKernel_divMin<<<dim_grid, dim_block, sharedMemBytes>>>(L_d, C_d, R_d, fsw_d, numL, numC, numR, numFsw, numHarmonics, worstAtten_d, thd_d);
    } else if(version == 4) {
        dim_grid.x = (totalCombinations + coarseFactor*blockSize - 1) / (coarseFactor*blockSize);
        dim_grid.y = dim_grid.z = 1;
        evaluateFilterKernel_coarsened<<<dim_grid, dim_block, sharedMemBytes>>>(L_d, C_d, R_d, fsw_d, numL, numC, numR, numFsw, numHarmonics, coarseFactor, worstAtten_d, thd_d);
    } else if(version == 5) {
        dim_grid.x = (totalCombinations + blockSize - 1) / blockSize;
        dim_grid.y = dim_grid.z = 1;
        evaluateFilterKernel_constant<<<dim_grid, dim_block>>>(L_d, C_d, numL, numC, numR, numFsw, numHarmonics, worstAtten_d, thd_d);
    } else {
        FATAL("Invalid kernel version selected");
    }

    cuda_ret = cudaDeviceSynchronize();
    if(cuda_ret != cudaSuccess) FATAL("Unable to launch/execute kernel");

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // D2H Data Transfer ////////////////////
    printf("Copying data from device to host..."); fflush(stdout);
    startTime(&timer);

    cuda_ret = cudaMemcpy(worstAtten_h, worstAtten_d, totalCombinations * sizeof(double), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");
    cuda_ret = cudaMemcpy(thd_h, thd_d, totalCombinations * sizeof(double), cudaMemcpyDeviceToHost);
    if(cuda_ret != cudaSuccess) FATAL("Unable to copy memory to host");

    cudaDeviceSynchronize();
    stopTime(&timer); printf("%f s\n", elapsedTime(timer));

    // THD Reduction ////////////////////
    printf("Performing global reduction..."); fflush(stdout);
    startTime(&timer);

    double worstGPU = 0.0, thdGPU = 0.0;
    reduceMin(worstAtten_d, totalCombinations, &worstGPU);
    reduceMax(thd_d, totalCombinations, &thdGPU);

    stopTime(&timer); printf("%f s\n", elapsedTime(timer));
    printf("    GPU global worst-case attenuation = %f dB, THD = %f\n", worstGPU, thdGPU);

    // Results Verification ////////////////////
    printf("Verifying results..."); fflush(stdout);

    verify(L_h, C_h, R_h, fsw_h, numL, numC, numR, numFsw, numHarmonics, worstAtten_h, thd_h, totalCombinations);

    cudaFree(L_d); 
    cudaFree(C_d); 
    cudaFree(R_d); 
    cudaFree(fsw_d);
    cudaFree(worstAtten_d); 
    cudaFree(thd_d);
    
    free(L_h); 
    free(C_h); 
    free(R_h); 
    free(fsw_h);
    free(worstAtten_h); 
    free(thd_h);

    return 0;
}