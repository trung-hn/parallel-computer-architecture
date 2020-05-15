/* Gaussian elimination code.
 * 
 * Author: Naga Kandasamy
 * Date of last update: April 22, 2020
 *
 * Student names(s): Trung Hoang
 * Date: 4/25/2020
 *
 * Compile as follows: 
 * gcc -o gauss_eliminate gauss_eliminate.c compute_gold.c -O3 -Wall -lpthread -lm
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "gauss_eliminate.h"

#define MIN_NUMBER 2
#define MAX_NUMBER 50

typedef struct thread_data_s
{
    int tid;
    int num_threads;
    int num_elements;
    int current_k;
    Matrix U;
} thread_data_t;

/* Function prototypes */
extern int compute_gold(float *, int);
Matrix allocate_matrix(int, int, int);
void gauss_eliminate_using_pthreads(Matrix, int);
int perform_simple_check(const Matrix);
void print_matrix(const Matrix);
float get_random_number(int, int);
int check_results(float *, float *, int, float);

int main(int argc, char **argv)
{
    if (argc < 2)
    {
        fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width and height of the square matrix\n");
        exit(EXIT_FAILURE);
    }

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    Matrix U;           /* Input matrix */
    Matrix U_reference; /* Upper triangular matrix computed by reference code */
    Matrix U_mt;        /* Upper triangular matrix computed by pthreads */

    fprintf(stderr, "Generating input matrices\n");
    srand(time(NULL));                                          /* Seed random number generator */
    U = allocate_matrix(matrix_size, matrix_size, 1);           /* Allocate and populate random square matrix */
    U_reference = allocate_matrix(matrix_size, matrix_size, 0); /* Allocate space for reference result */
    U_mt = allocate_matrix(matrix_size, matrix_size, 0);        /* Allocate space for multi-threaded result */

    /* Copy contents U matrix into U matrices */
    int i, j;
    for (i = 0; i < U.num_rows; i++)
    {
        for (j = 0; j < U.num_rows; j++)
        {
            U_reference.elements[U.num_rows * i + j] = U.elements[U.num_rows * i + j];
            U_mt.elements[U.num_rows * i + j] = U.elements[U.num_rows * i + j];
        }
    }

    fprintf(stderr, "\nPerforming gaussian elimination using reference code\n");
    struct timeval start, stop;
    gettimeofday(&start, NULL);

    int status = compute_gold(U_reference.elements, U.num_rows);

    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

    if (status < 0)
    {
        fprintf(stderr, "Failed to convert given matrix to upper triangular. Try again.\n");
        exit(EXIT_FAILURE);
    }

    status = perform_simple_check(U_reference); /* Check that principal diagonal elements are 1 */
    if (status < 0)
    {
        fprintf(stderr, "Upper triangular matrix is incorrect. Exiting.\n");
        exit(EXIT_FAILURE);
    }
    fprintf(stderr, "Single-threaded Gaussian elimination was successful.\n");

    /* FIXME: Perform Gaussian elimination using pthreads. 
     * The resulting upper triangular matrix should be returned in U_mt */
    fprintf(stderr, "\nPerforming gaussian elimination using pthreads\n");

    gettimeofday(&start, NULL);
    gauss_eliminate_using_pthreads(U_mt, num_threads);
    gettimeofday(&stop, NULL);
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

    /* Check if pthread result matches reference solution within specified tolerance */
    fprintf(stderr, "\nChecking results\n");
    int size = matrix_size * matrix_size;
    int res = check_results(U_reference.elements, U_mt.elements, size, 1e-6);
    fprintf(stderr, "TEST %s\n", (0 == res) ? "PASSED" : "FAILED");

    /* Free memory allocated for matrices */
    free(U.elements);
    free(U_reference.elements);
    free(U_mt.elements);

    exit(EXIT_SUCCESS);
}

// Striding
void *worker_division_step(void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;
    int tid = thread_data->tid;
    int stride = thread_data->num_threads;
    int k = thread_data->current_k;
    int num_elements = thread_data->num_elements;
    while (tid + 1 + k < thread_data->num_elements)
    {
        thread_data->U.elements[num_elements * k + (k + 1) + tid] = (float)thread_data->U.elements[num_elements * k + (k + 1) + tid] / thread_data->U.elements[num_elements * k + k];
        tid += stride;
    }
    pthread_exit(NULL);
    // From Gold
    // for (j = (k + 1); j < num_elements; j++) {
    //     U[num_elements * k + j] = (float)(U[num_elements * k + j] / U[num_elements * k + k]);	/* Division step */
    // }
}

// Striding
void *worker_elimination_step(void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;
    int tid = thread_data->tid;
    int stride = thread_data->num_threads;
    int k = thread_data->current_k;
    int num_elements = thread_data->num_elements;
    int j;
    // printf("%d %d\n", tid, k);
    while (tid + 1 + k < thread_data->num_elements)
    {
        for (j = (k + 1); j < num_elements; j++)
        {
            thread_data->U.elements[num_elements * ((k + 1) + tid) + j] -= thread_data->U.elements[num_elements * ((k + 1) + tid) + k] * thread_data->U.elements[num_elements * k + j];
        }
        thread_data->U.elements[num_elements * ((k + 1) + tid) + k] = 0;
        tid += stride;
    }
    pthread_exit(NULL);
    // From Gold
    //     for (i = (k + 1); i < num_elements; i++) {
    //         for (j = (k + 1); j < num_elements; j++)
    //             U[num_elements * i + j] = U[num_elements * i + j] - (U[num_elements * i + k] * U[num_elements * k + j]);
    //         U[num_elements * i + k] = 0;
    //     }
}

/* FIXME: Write code to perform gaussian elimination using pthreads */
void gauss_eliminate_using_pthreads(Matrix U, int num_threads)
{
    int num_rows = U.num_rows;
    int num_elements = num_rows;
    // printf("Row, Col: %d %d\n", num_columns, num_rows);
    // printf("Number of threads %d\n", num_threads);
    int i, k;

    pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
    thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
    for (k = 0; k < num_rows; k++)
    {
        for (i = 0; i < num_threads; i++)
        {
            thread_data[i].tid = i;
            thread_data[i].current_k = k;
            thread_data[i].num_threads = num_threads;
            thread_data[i].num_elements = num_elements;
            thread_data[i].U = U;
        }
        // Start Division Step
        for (i = 0; i < num_threads; i++)
            pthread_create(&thread_id[i], NULL, worker_division_step, (void *)&thread_data[i]);

        // Join Division Step
        for (i = 0; i < num_threads; i++)
            pthread_join(thread_id[i], NULL);
        // Change pivot to 1
        U.elements[num_elements * k + k] = 1; /* Set the principal diagonal entry in U to 1 */

        // Start Elimination Step, each thread handle 1 row
        for (i = 0; i < num_threads; i++){
            pthread_create(&thread_id[i], NULL, worker_elimination_step, (void *)&thread_data[i]);
        }
        // Join EliminationStep
        for (i = 0; i < num_threads; i++)
            pthread_join(thread_id[i], NULL);
    }
    free((void *)thread_data);
}

void print_matrix(Matrix U)
{
    int i,j;
    int num_elements = U.num_rows;
    for (i = 0; i < U.num_rows; i++)
    {
        for (j = 0; j < U.num_columns; j++)
        {
            printf("%.2f ", U.elements[num_elements * i + j]);
        }
        printf("\n");
    }
    printf("\n");
}

/* Check if results generated by single threaded and multi threaded versions match within tolerance */
int check_results(float *U, float *B, int size, float tolerance)
{
    int i;
    for (i = 0; i < size; i++)
        if (fabsf(U[i] - B[i]) > tolerance)
            return -1;
    return 0;
}

/* Allocate a matrix of dimensions height*width
 * If init == 0, initialize to all zeroes.  
 * If init == 1, perform random initialization. 
*/
Matrix allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;
    Matrix M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
    M.elements = (float *)malloc(size * sizeof(float));

    for (i = 0; i < size; i++)
    {
        if (init == 0)
            M.elements[i] = 0;
        else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
    }

    return M;
}

/* Return a random floating-point number between [min, max] */
float get_random_number(int min, int max)
{
    return (float)floor((double)(min + (max - min + 1) * ((float)rand() / (float)RAND_MAX)));
}

/* Perform simple check on upper triangular matrix if the principal diagonal elements are 1 */
int perform_simple_check(const Matrix M)
{
    int i;
    for (i = 0; i < M.num_rows; i++)
        if ((fabs(M.elements[M.num_rows * i + i] - 1.0)) > 1e-6)
            return -1;

    return 0;
}
