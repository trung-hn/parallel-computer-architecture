/* Blur filter. Device code. */

#ifndef _BLUR_FILTER_KERNEL_H_
#define _BLUR_FILTER_KERNEL_H_

#include "blur_filter.h"

__global__ void 
blur_filter_kernel (const float *in, float *out, int size)
{

    /* Obtain thread index within the thread block */
    int threadX = threadIdx.x;
    int threadY = threadIdx.y;

	/* Obtain block index within the grid */
	int blockX = blockIdx.x;
    int blockY = blockIdx.y;
    
    /* Find position in matrix */
	int col = blockDim.x * blockX + threadX;
    int row = blockDim.y * blockY + threadY;
    int curr_row, curr_col, i, j;
    long num_elements = size * size;
    
    int stride = blockDim.x * blockDim.y; // We have 32 * 32 = 1024 thread.
    long cnt = 0; // Make sure GPU stops if we make inf loop

    #ifdef DEBUG 
    printf("%d - %d - %d %d %d\n", blockDim.x, blockDim.y, BLOCK_SIZE, row, col);
    #endif

    for(int index = row * blockDim.x + col; index < num_elements; index += stride)
    {
        row = index / size;
        col = index % size;
        int num_neighbors = 0;
        float blur_value = 0.0;
        for (i = -BLUR_SIZE; i < (BLUR_SIZE + 1); i++) {
            for (j = -BLUR_SIZE; j < (BLUR_SIZE + 1); j++) {
                curr_row = row + i;
                curr_col = col + j;
                if ((curr_row > -1) && (curr_row < size) && (curr_col > -1) && (curr_col < size)) {
                    blur_value += in[curr_row * size + curr_col];
                    num_neighbors += 1;
                }
            }
        }

        cnt += 1;
        if (cnt == 10000000)
            break;
        /* Write averaged blurred value out */
        out[row * size + col] = blur_value/num_neighbors;
    }
    return;
}

#endif /* _BLUR_FILTER_KERNEL_H_ */
