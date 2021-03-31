#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(float *A, float *x, float *B, int matrix_size)
{
    /* Obtain thread index within the thread block */
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	/* Obtain block index within the grid */
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	/* Find position in matrix */
	int column_number = blockDim.x * blockX + threadX;
    int row_number = blockDim.y * blockY + threadY;
    
    




    return;
}

__global__ void jacobi_iteration_kernel_optimized()
{
    return;
}

