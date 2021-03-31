/* Host code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Build as follws: make clean && make

 * Author: Naga Kandasamy
 * Date modified: May 21, 2020
*/

#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "jacobi_iteration.h"

/* Include the kernel code */
#include "jacobi_iteration_kernel.cu"

/* Uncomment the line below if you want the code to spit out debug information. */ 
// #define DEBUG

int main(int argc, char **argv) 
{
	if (argc > 1) {
		printf("This program accepts no arguments\n");
		exit(EXIT_FAILURE);
	}
    
    /* Allocate memory for the input and output images */
    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
	matrix_t gpu_naive_solution_x;  /* Solution computed by naive kernel */
    matrix_t gpu_opt_solution_x;    /* Solution computed by optimized kernel */

	/* Initialize the random number generator */
	srand(time(NULL));

	/* Generate diagonally dominant matrix */ 
    printf("\nGenerating %d x %d system\n", MATRIX_SIZE, MATRIX_SIZE);
	A = create_diagonally_dominant_matrix(MATRIX_SIZE, MATRIX_SIZE);
	if (A.elements == NULL) {
        printf("Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create the other vectors */
    B = allocate_matrix_on_host(MATRIX_SIZE, 1, 1);
	reference_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
	gpu_naive_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);
    gpu_opt_solution_x = allocate_matrix_on_host(MATRIX_SIZE, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution on CPU */
    printf("\nPerforming Jacobi iteration on the CPU\n");
    struct timeval start, stop;	
	gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "Gold execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
    
    
    /* Compute Jacobi solution on device. Solutions are returned 
       in gpu_naive_solution_x and gpu_opt_solution_x. */
    printf("\nPerforming Jacobi iteration on device\n");
    compute_on_device(A, gpu_naive_solution_x, gpu_opt_solution_x, B);
    

#ifdef DEBUG
	print_matrix(gpu_naive_solution_x);
#endif

    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(gpu_naive_solution_x.elements);
    free(gpu_opt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}

void naive_method(const matrix_t A, matrix_t gpu_naive_sol_x, const matrix_t B)
{
    // Malloc on CUDA
    matrix_t Ad = allocate_matrix_on_device(A);
    copy_matrix_to_device(Ad, A);
    matrix_t Bd = allocate_matrix_on_device(B);
    copy_matrix_to_device(Bd, B);
    matrix_t Xd = allocate_matrix_on_device(gpu_naive_sol_x);
    matrix_t new_Xd = allocate_matrix_on_device(gpu_naive_sol_x);

    double *ssdd;
    cudaMalloc(&ssdd, sizeof(double));
    
    double ssd, mse;
    unsigned int done = 0;
    unsigned int num_iter = 0;
    
    dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);          /* Set number of threads in thread block */
    dim3 grid(NUMBLOCKS,1);
    long num_threads = NUMBLOCKS * THREAD_BLOCK_SIZE;

    // Start timer
    struct timeval start, stop;	
    gettimeofday(&start, NULL);
    
    while (!done) {
        cudaMemset(ssdd, 0.0, sizeof(double));

        // calculate new_x
	    jacobi_iteration_kernel_naive<<<grid, thread_block>>>(Ad.elements, Xd.elements, new_Xd.elements, Bd.elements, ssdd, num_threads);
        cudaDeviceSynchronize();

        // Get ssd from kernel
        cudaMemcpy(&ssd, ssdd, sizeof(double), cudaMemcpyDeviceToHost);
        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
        if (mse <= THRESHOLD ||  num_iter > 20000)
            done = 1;
    }

    // Stop timer
    gettimeofday(&stop, NULL);
    printf("\nConvergence achieved after %d iterations \n", num_iter);
    fprintf(stderr, "Naive execution time = %fs\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

    // Get from GPU
    copy_matrix_from_device(gpu_naive_sol_x, Xd);
    cudaFree(ssdd);
}

void optimal_method(const matrix_t A, matrix_t gpu_opt_sol_x, const matrix_t B)
{
    // Malloc on CUDA
    matrix_t Ad = allocate_matrix_on_device(A);
    copy_matrix_to_device(Ad, A);
    matrix_t Bd = allocate_matrix_on_device(B);
    copy_matrix_to_device(Bd, B);
    matrix_t Xd = allocate_matrix_on_device(gpu_opt_sol_x);
    matrix_t new_Xd = allocate_matrix_on_device(gpu_opt_sol_x);

    double *ssdd;
    cudaMalloc(&ssdd, sizeof(double));
    
    double ssd, mse;
    unsigned int done = 0;
    unsigned int num_iter = 0;
    
    dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);          /* Set number of threads in thread block */
    dim3 grid(NUMBLOCKS,1);
    long num_threads = NUMBLOCKS * THREAD_BLOCK_SIZE;

    // Start timer
    struct timeval start, stop;	
    gettimeofday(&start, NULL);
    int ping = 1;
    while (!done) {
        cudaMemset(ssdd, 0.0, sizeof(double));
        // calculate new_x
        if (ping)
            jacobi_iteration_kernel_optimized<<<grid, thread_block>>>(Ad.elements, Xd.elements, new_Xd.elements, Bd.elements, ssdd, num_threads);
        else
            jacobi_iteration_kernel_optimized<<<grid, thread_block>>>(Ad.elements, new_Xd.elements, Xd.elements, Bd.elements, ssdd, num_threads);
        cudaDeviceSynchronize();
        ping = 1 - ping;
        
        // Get ssd from kernel
        cudaMemcpy(&ssd, ssdd, sizeof(double), cudaMemcpyDeviceToHost);
        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
        if (mse <= THRESHOLD ||  num_iter > 20000)
            done = 1;
    }

    // Stop timer
    gettimeofday(&stop, NULL);
    printf("\nConvergence achieved after %d iterations \n", num_iter);
    fprintf(stderr, "Optimal execution time = %fs\n\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec)/(float)1000000));

    // Get from GPU
    copy_matrix_from_device(gpu_opt_sol_x, Xd);
    cudaFree(ssdd);
}

/* FIXME: Complete this function to perform Jacobi calculation on device */
void compute_on_device(const matrix_t A, matrix_t gpu_naive_sol_x, matrix_t gpu_opt_sol_x, const matrix_t B)
{
    naive_method(A, gpu_naive_sol_x, B);
    display_jacobi_solution(A, gpu_naive_sol_x, B); /* Display statistics */

    // Transpose A to make it coalesced
    for (int i = 0; i < A.num_rows; i++){
        for (int j = i + 1; j < A.num_rows; j++){
            int temp = A.elements[i * A.num_rows + j];
            A.elements[i * A.num_rows + j] = A.elements[i * A.num_rows + j + (A.num_rows - 1) * (j - i)];
            A.elements[i * A.num_rows + j + (A.num_rows - 1) * (j - i)] = temp;
        }
    }
    optimal_method(A, gpu_opt_sol_x, B);
    // Transpose A again to calculate score
    for (int i = 0; i < A.num_rows; i++){
        for (int j = i + 1; j < A.num_rows; j++){
            int temp = A.elements[i * A.num_rows + j];
            A.elements[i * A.num_rows + j] = A.elements[i * A.num_rows + j + (A.num_rows - 1) * (j - i)];
            A.elements[i * A.num_rows + j + (A.num_rows - 1) * (j - i)] = temp;
        }
    }
    display_jacobi_solution(A, gpu_opt_sol_x, B); 
}



/* Allocate matrix on the device of same size as M */
matrix_t allocate_matrix_on_device(const matrix_t M)
{
    matrix_t Mdevice = M;
    int size = M.num_rows * M.num_columns * sizeof(float);
    cudaMalloc((void **)&Mdevice.elements, size);
    return Mdevice;
}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix_on_host(int num_rows, int num_columns, int init)
{	
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (unsigned int i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Copy matrix to device */
void copy_matrix_to_device(matrix_t Mdevice, const matrix_t Mhost)
{
    int size = Mhost.num_rows * Mhost.num_columns * sizeof(float);
    Mdevice.num_rows = Mhost.num_rows;
    Mdevice.num_columns = Mhost.num_columns;
    cudaMemcpy(Mdevice.elements, Mhost.elements, size, cudaMemcpyHostToDevice);
    return;
}

/* Copy matrix from device to host */
void copy_matrix_from_device(matrix_t Mhost, const matrix_t Mdevice)
{
    int size = Mdevice.num_rows * Mdevice.num_columns * sizeof(float);
    cudaMemcpy(Mhost.elements, Mdevice.elements, size, cudaMemcpyDeviceToHost);
    return;
}

/* Prints the matrix out to screen */
void print_matrix(const matrix_t M)
{
	for (unsigned int i = 0; i < M.num_rows; i++) {
        for (unsigned int j = 0; j < M.num_columns; j++) {
			printf("%f ", M.elements[i * M.num_columns + j]);
        }
		
        printf("\n");
	} 
	
    printf("\n");
    return;
}

/* Returns a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check for errors in kernel execution */
void check_CUDA_error(const char *msg)
{
	cudaError_t err = cudaGetLastError();
	if ( cudaSuccess != err) {
		printf("CUDA ERROR: %s (%s).\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}	
    
    return;    
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix(unsigned int num_rows, unsigned int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	unsigned int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));
    if (M.elements == NULL)
        return M;

	/* Create a matrix with random numbers between [-.5 and .5] */
    unsigned int i, j;
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number (MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
	for (i = 0; i < num_rows; i++) {
		float row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    return M;
}

