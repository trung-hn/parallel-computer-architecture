/* Host code that implements a  separable convolution filter of a 
 * 2D signal with a gaussian kernel.
 * 
 * Author: Naga Kandasamy
 * Date modified: May 26, 2020
 */

 #include <stdlib.h>
 #include <stdio.h>
 #include <string.h>
 #include <math.h>
 #include <time.h>
 #include <sys/time.h>
 
 extern "C" void compute_gold(float *, float *, int, int, int);
 extern "C" float *create_kernel(float, int);
 void check_for_error(const char *);
 void print_kernel(float *, int);
 void print_matrix(float *, int, int);
 
 /* Width of convolution kernel */
 #define HALF_WIDTH 8
 #define WIDTH 17
 #define COEFF 10
 
 /* Uncomment line below to spit out debug information */
 //#define DEBUG
 
 /* Include device code */
 
 __constant__ float kernel_c[WIDTH]; 
 
 #include "separable_convolution_kernel.cu"
 
 #define THREAD_BLOCK_SIZE 256
 
 
 void compute_on_device_naive(float *gpu_result, float *matrix_c,\
    float *kernel, int num_cols,\
    int num_rows, int half_width)
{
struct timeval start, stop;
float *in_dev = NULL;
float *out_dev = NULL;
float *out2_dev = NULL;
float *kernel_dev = NULL;

int num_elements = num_rows * num_cols;
int width = 2 * half_width + 1;

cudaMalloc((void **)&in_dev, num_elements * sizeof(float));
cudaMemcpy(in_dev, matrix_c, num_elements * sizeof(float), cudaMemcpyHostToDevice);

cudaMalloc((void **)&kernel_dev, width * sizeof(float));
cudaMemcpy(kernel_dev, kernel, width * sizeof(float), cudaMemcpyHostToDevice);

cudaMalloc((void **)&out_dev, num_elements * sizeof(float));
cudaMalloc((void **)&out2_dev, num_elements * sizeof(float));

int num_thread_blocks = ceil((float)num_elements/(float)(THREAD_BLOCK_SIZE));
dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);
fprintf(stderr, "Setting up a (%d x 1) execution grid\n", num_thread_blocks);
dim3 grid(num_thread_blocks, 1);


fprintf(stderr, "2D Convolution with Naive\n");
fprintf(stderr, "Convolving over Rows\n"); 
gettimeofday(&start, NULL);
convolve_rows_kernel_naive<<<grid, thread_block>>>(out_dev, in_dev, kernel_dev, num_cols, num_rows, half_width);
cudaDeviceSynchronize();

check_for_error("KERNEL FAILURE");

fprintf(stderr, "Convolving over Columns\n");
convolve_columns_kernel_naive<<<grid, thread_block>>>(out2_dev, out_dev, kernel_dev, num_cols, num_rows, half_width);
cudaDeviceSynchronize();
gettimeofday(&stop, NULL);
fprintf(stderr, "Naive run time = %0.4f s\n", (float)(stop.tv_sec - start.tv_sec\
            + (stop.tv_usec - start.tv_usec) / (float)1000000));

check_for_error("KERNEL FAILURE");

cudaMemcpy(gpu_result, out2_dev, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef DEBUG
print_matrix(gpu_result, num_cols, num_rows); 
#endif

/* Free memory on GPU */	  
cudaFree(in_dev); 
cudaFree(out_dev); 
cudaFree(out2_dev);
cudaFree(kernel_dev);

return;
}


/* Function to compute the convolution on the GPU */
void compute_on_device_opt(float *gpu_result, float *matrix_c,\
    float *kernel, int num_cols,\
    int num_rows, int half_width)
{
struct timeval start, stop;
float *in_dev = NULL;
float *out_dev = NULL;
float *out2_dev = NULL;
float *kernel_dev = NULL;

int num_elements = num_rows * num_cols;
int width = 2 * half_width + 1;

cudaMalloc((void **)&in_dev, num_elements * sizeof(float));
cudaMemcpy(in_dev, matrix_c, num_elements * sizeof(float), cudaMemcpyHostToDevice);

cudaMalloc((void **)&kernel_dev, width * sizeof(float));
cudaMemcpy(kernel_dev, kernel, width * sizeof(float), cudaMemcpyHostToDevice);

cudaMalloc((void **)&out_dev, num_elements * sizeof(float));
cudaMalloc((void **)&out2_dev, num_elements * sizeof(float));

int num_thread_blocks = ceil((float)num_elements/(float)(THREAD_BLOCK_SIZE));
dim3 thread_block(THREAD_BLOCK_SIZE, 1, 1);
fprintf(stderr, "Setting up a (%d x 1) execution grid\n", num_thread_blocks);
dim3 grid(num_thread_blocks, 1, 1);

cudaMemcpyToSymbol(kernel_c, kernel_dev, width * sizeof(float));

fprintf(stderr, "2D Convolution with Optimized\n");
fprintf(stderr, "Convolving over Rows\n"); 
gettimeofday(&start, NULL);
convolve_rows_kernel_optimized<<<grid, thread_block>>>(out_dev, in_dev, num_cols, num_rows, half_width);
cudaDeviceSynchronize();

check_for_error("KERNEL FAILURE");

fprintf(stderr, "Convolving over Columns\n");
convolve_columns_kernel_optimized<<<grid, thread_block>>>(out2_dev, out_dev, num_cols, num_rows, half_width);
cudaDeviceSynchronize();
gettimeofday(&stop, NULL);
fprintf(stderr, "Optimized run time = %0.4f s\n", (float)(stop.tv_sec - start.tv_sec\
            + (stop.tv_usec - start.tv_usec) / (float)1000000));
check_for_error("KERNEL FAILURE");

cudaMemcpy(gpu_result, out2_dev, num_elements * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef DEBUG
print_matrix(gpu_result, num_cols, num_rows); 
#endif

/* Free memory on GPU */	  
cudaFree(in_dev); 
cudaFree(out_dev); 
cudaFree(out2_dev);
cudaFree(kernel_dev);

return;
}

 int main(int argc, const char **argv)
 {
     if (argc < 3) {
         printf("Usage: %s num-rows num-columns\n", argv[0]);
         printf("num-rows: height of the matrix\n");
         printf("num-columns: width of the matrix\n");
         exit(EXIT_FAILURE);
     }
 
     int num_rows = atoi(argv[1]);
     int num_cols = atoi(argv[2]);
 
     struct timeval start, stop;
 
     /* Create input matrix */
     int num_elements = num_rows * num_cols;
     printf("Creating input matrix of %d x %d\n", num_rows, num_cols);
     float *matrix_a = (float *)malloc(sizeof(float) * num_elements);
     float *matrix_c = (float *)malloc(sizeof(float) * num_elements);
     
     srand(time(NULL));
     int i;
     for (i = 0; i < num_elements; i++) {
         matrix_a[i] = rand()/(float)RAND_MAX;			 
         matrix_c[i] = matrix_a[i]; /* Copy contents of matrix_a into matrix_c */
     }
      
     /* Create Gaussian kernel */	  
     float *gaussian_kernel = create_kernel((float)COEFF, HALF_WIDTH);	
 #ifdef DEBUG
     print_kernel(gaussian_kernel, HALF_WIDTH); 
 #endif
       
     /* Convolve matrix along rows and columns. 
        The result is stored in matrix_a, thereby overwriting the 
        original contents of matrix_a.		
      */
     printf("\nConvolving the matrix on the CPU\n");	  
     gettimeofday(&start, NULL);
     compute_gold(matrix_a, gaussian_kernel, num_cols,\
                   num_rows, HALF_WIDTH);
     gettimeofday(&stop, NULL);
     fprintf(stderr, "CPU run time = %0.4f s\n", (float)(stop.tv_sec - start.tv_sec\
                 + (stop.tv_usec - start.tv_usec) / (float)1000000));
 
 
 #ifdef DEBUG	 
     print_matrix(matrix_a, num_cols, num_rows);
 #endif
   
     float *gpu_result_naive = (float *)malloc(sizeof(float) * num_elements);
     
     /* Convolve matrix along rows and columns using GPU_Naive
      */
     printf("\nConvolving matrix on the GPU Naive\n");
     compute_on_device_naive(gpu_result_naive, matrix_c, gaussian_kernel, num_cols,\
                        num_rows, HALF_WIDTH);
       
 #ifdef DEBUG	 
     printf("\n");
     print_matrix(gpu_result_naive, num_cols, num_rows);
 #endif
 
 
     printf("\nComparing CPU and GPU-Naive results\n");
     float sum_delta = 0, sum_ref = 0;
     for (i = 0; i < num_elements; i++) {
         sum_delta += fabsf(matrix_a[i] - gpu_result_naive[i]);
         sum_ref   += fabsf(matrix_a[i]);
     }
         
     float L1norm = sum_delta / sum_ref;
     float eps = 1e-6;
     printf("L1 norm: %E\n", L1norm);
     printf((L1norm < eps) ? "TEST PASSED\n" : "TEST FAILED\n");
 
     float *gpu_result_opt = (float *)malloc(sizeof(float) * num_elements);
 
     /* Convolve matrix along rows and columns using GPU_Naive
      */
     printf("\nConvolving matrix on the GPU Optimized\n");
     compute_on_device_opt(gpu_result_opt, matrix_c, gaussian_kernel, num_cols,\
                        num_rows, HALF_WIDTH);
       
 #ifdef DEBUG	 
     printf("\n");
     print_matrix(gpu_result_opt, num_cols, num_rows);
 #endif
 
 
     printf("\nComparing CPU and GPU-Opt results\n");
     sum_delta = 0;
     sum_ref = 0;
     for (i = 0; i < num_elements; i++) {
         sum_delta += fabsf(matrix_a[i] - gpu_result_opt[i]);
         sum_ref   += fabsf(matrix_a[i]);
     }
         
     L1norm = sum_delta / sum_ref;
     eps = 1e-6;
     printf("L1 norm: %E\n", L1norm);
     printf((L1norm < eps) ? "TEST PASSED\n" : "TEST FAILED\n");
 
 
     free(matrix_a);
     free(matrix_c);
     free(gpu_result_naive);
     free(gpu_result_opt);
     free(gaussian_kernel);
 
     exit(EXIT_SUCCESS);
 }
 
 
 /* Check for errors reported by the CUDA run time */
 void check_for_error(const char *msg)
 {
     cudaError_t err = cudaGetLastError();
     if (cudaSuccess != err) {
         printf("CUDA ERROR: %s (%s)\n", msg, cudaGetErrorString(err));
         exit(EXIT_FAILURE);
     }
 
     return;
 } 
 
 /* Print convolution kernel */
 void print_kernel(float *kernel, int half_width)
 {
     int i, j = 0;
     for (i = -half_width; i <= half_width; i++) {
         printf("%0.2f ", kernel[j]);
         j++;
     }
 
     printf("\n");
     return;
 }
 
 /* Print matrix */
 void print_matrix(float *matrix, int num_cols, int num_rows)
 {
     int i,  j;
     float element;
     for (i = 0; i < num_rows; i++) {
         for (j = 0; j < num_cols; j++){
             element = matrix[i * num_cols + j];
             printf("%0.2f ", element);
         }
         printf("\n");
     }
 
     return;
 }