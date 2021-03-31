#include "jacobi_iteration.h"

/* FIXME: Write the device kernels to solve the Jacobi iterations */


__global__ void jacobi_iteration_kernel_naive(float *A, float *x, float *new_x,float *B, double *ssd, long num_threads)
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
	int stride = num_threads;
	int thread_id = row_number * blockDim.x + column_number;
	// printf("%d %d %d %d %d\n", thread_id, blockDim.x,  blockDim.y, row_number, column_number);

	int i, j;

	for (i = thread_id; i < MATRIX_SIZE; i += stride) {
		double sum = -A[i * MATRIX_SIZE + i] * x[i];
		for (j = 0; j < MATRIX_SIZE; j++) {
			sum += A[i * MATRIX_SIZE + j] * x[j];
		}
		/* Update values for the unkowns for the current row. */
		new_x[i] = (B[i] - sum)/A[i * MATRIX_SIZE + i];
	}
	__syncthreads (); /* Wait for all threads in thread block to finish */

	for (i = thread_id; i < MATRIX_SIZE; i += stride) {
		// *ssd += (new_x[i] - x[i]) * (new_x[i] - x[i]);
		atomicAdd(ssd, (new_x[i] - x[i]) * (new_x[i] - x[i]));
		x[i] = new_x[i];
	}
    return;
}

__global__ void jacobi_iteration_kernel_optimized(float *A, float *x, float *new_x,float *B, double *ssd, long num_threads)
{
	__shared__ float ssdSub[MATRIX_SIZE];

    /* Obtain thread index within the thread block */
	int threadX = threadIdx.x;
	int threadY = threadIdx.y;

	/* Obtain block index within the grid */
	int blockX = blockIdx.x;
	int blockY = blockIdx.y;

	/* Find position in matrix */
	int column_number = blockDim.x * blockX + threadX;
	int row_number = blockDim.y * blockY + threadY;
	int stride = num_threads;
	int thread_id = row_number * blockDim.x + column_number;
	// printf("%d %d %d %d %d\n", thread_id, blockDim.x,  blockDim.y, row_number, column_number);

	int i, j;

	for (i = thread_id; i < MATRIX_SIZE; i += stride) {
		double sum = -A[i * MATRIX_SIZE + i] * x[i];
		for (j = 0; j < MATRIX_SIZE; j++) {
			sum += A[i + j * MATRIX_SIZE ] * x[j];
		}
	   
		/* Update values for the unkowns for the current row. */
		new_x[i] = (B[i] - sum)/A[i * MATRIX_SIZE + i];
	}
	__syncthreads ();
	
	for (i = thread_id; i < MATRIX_SIZE; i += stride) {
		ssdSub[i] = (new_x[i] - x[i]) * (new_x[i] - x[i]);
	}
	
	for (i = thread_id; i < MATRIX_SIZE; i += stride) {
		atomicAdd(ssd, ssdSub[i]);
	}

    return;
}

