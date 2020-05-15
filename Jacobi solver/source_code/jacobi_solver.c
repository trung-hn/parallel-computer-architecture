/* Code for the Jacobi method of solving a system of linear equations 
 * by iteration.

 * Author: Naga Kandasamy
 * Date modified: April 22, 2020
 *
 * Compile as follows:
 * gcc -o jacobi_solver jacobi_solver.c compute_gold.c -Wall -O3 -lpthread -lm 
*/

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <semaphore.h>
#include <math.h>
#include <pthread.h>
#include "jacobi_solver.h"

/* Uncomment the line below to spit out debug information */ 
/* #define DEBUG */


/* Structure that defines the barrier */
typedef struct barrier_s
{
    sem_t counter_sem; /* Protects access to the counter */
    sem_t barrier_sem; /* Signals that barrier is safe to cross */
    int counter;       /* The value itself */
} barrier_t;

barrier_t barrier;
barrier_t barrier2;

typedef struct thread_data_s
{
    int tid;
    int num_threads;
    int num_rows;
    matrix_t A;
    matrix_t B;
	matrix_t mt_sol_x;
	matrix_t new_x;
	int *done;
} thread_data_t;

void barrier_sync(barrier_t *, int, int);

int main(int argc, char **argv) 
{
	if (argc < 2) {
		fprintf(stderr, "Usage: %s matrix-size\n", argv[0]);
        fprintf(stderr, "matrix-size: width of the square matrix\n");
		exit(EXIT_FAILURE);
	}

    int matrix_size = atoi(argv[1]);
    int num_threads = atoi(argv[2]);

    matrix_t  A;                    /* N x N constant matrix */
	matrix_t  B;                    /* N x 1 b matrix */
	matrix_t reference_x;           /* Reference solution */ 
    matrix_t mt_solution_x;         /* Solution computed by pthread code */

	/* Initialize the barrier data structure */
    barrier.counter = 0;
    sem_init(&barrier.counter_sem, 0, 1); /* Initialize semaphore protecting the counter to unlocked */
    sem_init(&barrier.barrier_sem, 0, 0); /* Initialize semaphore protecting the barrier to locked */
    
	barrier2.counter = 0;
    sem_init(&barrier2.counter_sem, 0, 1); /* Initialize semaphore protecting the counter to unlocked */
    sem_init(&barrier2.barrier_sem, 0, 0); /* Initialize semaphore protecting the barrier to locked */

	/* Generate diagonally dominant matrix */
    fprintf(stderr, "\nCreating input matrices\n");
	srand(time(NULL));
	A = create_diagonally_dominant_matrix(matrix_size, matrix_size);
	if (A.elements == NULL) {
        fprintf(stderr, "Error creating matrix\n");
        exit(EXIT_FAILURE);
	}
	
    /* Create other matrices */
    B = allocate_matrix(matrix_size, 1, 1);
	reference_x = allocate_matrix(matrix_size, 1, 0);
	mt_solution_x = allocate_matrix(matrix_size, 1, 0);

#ifdef DEBUG
	print_matrix(A);
	print_matrix(B);
	print_matrix(reference_x);
#endif

    /* Compute Jacobi solution using reference code */
	fprintf(stderr, "Generating solution using reference code\n");
    int max_iter = 100000; /* Maximum number of iterations to run */
    struct timeval start, stop;
	gettimeofday(&start, NULL);
    compute_gold(A, reference_x, B, max_iter);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, reference_x, B); /* Display statistics */
	
    fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));

	/* Compute the Jacobi solution using pthreads. 
     * Solutions are returned in mt_solution_x.
     * */
    fprintf(stderr, "\nPerforming Jacobi iteration using pthreads\n");
	gettimeofday(&start, NULL);
	compute_using_pthreads(A, mt_solution_x, B, num_threads);
    gettimeofday(&stop, NULL);
    display_jacobi_solution(A, mt_solution_x, B); /* Display statistics */
    
	fprintf(stderr, "CPU run time = %0.2f s\n", (float)(stop.tv_sec - start.tv_sec + (stop.tv_usec - start.tv_usec) / (float)1000000));
    
    free(A.elements); 
	free(B.elements); 
	free(reference_x.elements); 
	free(mt_solution_x.elements);
	
    exit(EXIT_SUCCESS);
}

// Striding
void *worker_job(void *args)
{
    thread_data_t *thread_data = (thread_data_t *)args;
	int tid = thread_data->tid;
	int stride = thread_data->num_threads;
	int num_rows = thread_data->num_rows;
	int num_cols = thread_data->num_rows;
	matrix_t new_x = thread_data->new_x;
	matrix_t x = thread_data->mt_sol_x;
	matrix_t A = thread_data->A;
	matrix_t B = thread_data->B;
	int i, j;
    
    while (! *thread_data->done) {

		i = thread_data->tid;
		while (i < num_rows){

            double sum = -A.elements[i * num_cols + i] * x.elements[i];
            for (j = 0; j < num_cols; j++) {
				sum += A.elements[i * num_cols + j] * x.elements[j];
            }
           
            /* Update values for the unkowns for the current row. */
            new_x.elements[i] = (B.elements[i] - sum)/A.elements[i * num_cols + i];

			// Striding
			i += stride;
        }
		
		// Let main() knows that new_x is finished
        barrier_sync(&barrier, tid, thread_data->num_threads + 1);

		// Wait until next iteration
        barrier_sync(&barrier2, tid, thread_data->num_threads + 1);
    }
	return 0;
}

/* FIXME: Complete this function to perform the Jacobi calculation using pthreads. 
 * Result must be placed in mt_sol_x. */
void compute_using_pthreads(const matrix_t A, matrix_t mt_sol_x, const matrix_t B, int num_threads)
{
	// int num_threads = 8;
	int num_rows = A.num_rows;
	int i;
	
	int max_iter = 100000; /* Maximum number of iterations to run */
	int done = 0;
    double ssd, mse;
    int num_iter = 0;

	matrix_t new_x = allocate_matrix(num_rows, 1, 0);

	pthread_t *thread_id = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	thread_data_t *thread_data = (thread_data_t *)malloc(sizeof(thread_data_t) * num_threads);
	for (i = 0; i < num_threads; i++)
	{
		thread_data[i].tid = i;
		thread_data[i].num_threads = num_threads;
		thread_data[i].num_rows = num_rows;
		thread_data[i].mt_sol_x = mt_sol_x;
		thread_data[i].new_x = new_x;
		thread_data[i].A = A;
		thread_data[i].B = B;
		thread_data[i].done = &done;
	}
	
	// Fork point
    for (i = 0; i < num_threads; i++){
        pthread_create(&thread_id[i], NULL, worker_job, (void *)&thread_data[i]);
	}


	while (!done) {
		// Wait until threads finish calculating new_x
        barrier_sync(&barrier, thread_data->num_threads + 1, thread_data->num_threads + 1);
        
		ssd = 0.0; 
        /* Check for convergence and update the unknowns. */
        for (i = 0; i < num_rows; i++) {
            ssd += (new_x.elements[i] - mt_sol_x.elements[i]) * (new_x.elements[i] - mt_sol_x.elements[i]);
            mt_sol_x.elements[i] = new_x.elements[i];
        }
        num_iter++;
        mse = sqrt(ssd); /* Mean squared error. */
        
        if ((mse <= THRESHOLD) || (num_iter == max_iter))
            done = 1;

		// Notify threads to start next iteration
        barrier_sync(&barrier2, thread_data->num_threads + 1, thread_data->num_threads + 1);
    }

    // Join point
    for (i = 0; i < num_threads; i++){
        pthread_join(thread_id[i], NULL);
	}
    free((void *)thread_data);

}

/* Allocate a matrix of dimensions height * width.
   If init == 0, initialize to all zeroes.  
   If init == 1, perform random initialization.
*/
matrix_t allocate_matrix(int num_rows, int num_columns, int init)
{
    int i;    
    matrix_t M;
    M.num_columns = num_columns;
    M.num_rows = num_rows;
    int size = M.num_rows * M.num_columns;
		
	M.elements = (float *)malloc(size * sizeof(float));
	for (i = 0; i < size; i++) {
		if (init == 0) 
            M.elements[i] = 0; 
		else
            M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	}
    
    return M;
}	

/* Print matrix to screen */
void print_matrix(const matrix_t M)
{
    int i, j;
	for (i = 0; i < M.num_rows; i++) {
        for (j = 0; j < M.num_columns; j++) {
			fprintf(stderr, "%f ", M.elements[i * M.num_columns + j]);
        }
		
        fprintf(stderr, "\n");
	} 
	
    fprintf(stderr, "\n");
    return;
}

/* Return a floating-point value between [min, max] */
float get_random_number(int min, int max)
{
    float r = rand ()/(float)RAND_MAX;
	return (float)floor((double)(min + (max - min + 1) * r));
}

/* Check if matrix is diagonally dominant */
int check_if_diagonal_dominant(const matrix_t M)
{
    int i, j;
	float diag_element;
	float sum;
	for (i = 0; i < M.num_rows; i++) {
		sum = 0.0; 
		diag_element = M.elements[i * M.num_rows + i];
		for (j = 0; j < M.num_columns; j++) {
			if (i != j)
				sum += abs(M.elements[i * M.num_rows + j]);
		}
		
        if (diag_element <= sum)
			return -1;
	}

	return 0;
}

/* Create diagonally dominant matrix */
matrix_t create_diagonally_dominant_matrix (int num_rows, int num_columns)
{
	matrix_t M;
	M.num_columns = num_columns;
	M.num_rows = num_rows; 
	int size = M.num_rows * M.num_columns;
	M.elements = (float *)malloc(size * sizeof(float));

    int i, j;
	fprintf(stderr, "Generating %d x %d matrix with numbers between [-.5, .5]\n", num_rows, num_columns);
	for (i = 0; i < size; i++)
        M.elements[i] = get_random_number(MIN_NUMBER, MAX_NUMBER);
	
	/* Make diagonal entries large with respect to the entries on each row. */
    float row_sum;
	for (i = 0; i < num_rows; i++) {
		row_sum = 0.0;		
		for (j = 0; j < num_columns; j++) {
			row_sum += fabs(M.elements[i * M.num_rows + j]);
		}
		
        M.elements[i * M.num_rows + i] = 0.5 + row_sum;
	}

    /* Check if matrix is diagonal dominant */
	if (check_if_diagonal_dominant(M) < 0) {
		free(M.elements);
		M.elements = NULL;
	}
	
    return M;
}

/* Barrier synchronization implementation */
void barrier_sync(barrier_t *barrier, int tid, int num_threads)
{
    int i;

    sem_wait(&(barrier->counter_sem));
    /* Check if all threads before us, that is num_threads - 1 threads have reached this point. */
    if (barrier->counter == (num_threads - 1))
    {
        barrier->counter = 0; /* Reset counter value */
        sem_post(&(barrier->counter_sem));
        /* Signal blocked threads that it is now safe to cross the barrier */
        // printf("Thread number %d is signalling other threads to proceed\n", tid);
        for (i = 0; i < (num_threads - 1); i++)
            sem_post(&(barrier->barrier_sem));
    }
    else
    { /* There are threads behind us */
        barrier->counter++;
        sem_post(&(barrier->counter_sem));
        sem_wait(&(barrier->barrier_sem)); /* Block on the barrier semaphore */
    }

    return;
}
