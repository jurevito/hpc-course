#include <CL/cl.h>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "mtx_sparse.h"

/*
size: 217918 x 217918, nonzero: 5926171, max elems in row: 180
Errors: 0(CSR_seq), 0(ELL_seq), 0(CSR_cl), 0(ELL_cl)
Times: 0.847000(COO_seq), 0.442000(CSR_seq), 6.826000(ELL_seq)
Times: 0.075000(CSR_cl), 0.115000(ELL_cl)

size: 217918 x 217918, nonzero: 5926171, max elems in row: 180
Errors: 0(CSR_seq), 0(ELL_seq), 0(CSR_cl), 0(ELL_cl)
Times: 0.847000(COO_seq), 0.442000(CSR_seq), 6.826000(ELL_seq)
Times: 0.075000(CSR_cl), 0.115000(ELL_cl)
       0.063000

srun --reservation=fri gcc -O2 -fopenmp -lm -lOpenCL main.c mtx_sparse.c -o main


time: 0.098734 repetitions: 100
Number of errors: 144846

size: 217918 x 217918, nonzero: 5926171, max elems in row: 180
Errors: 0(CSR_seq), 0(ELL_seq), 0(CSR_cl), 0(ELL_cl)
Times: 1.389209(COO_seq), 0.649234(CSR_seq), 6.710061(ELL_seq)
Times: 0.358190(CSR_cl), 0.325047(ELL_cl)

threads_per_row = 8 work_group_size = 64  
time: 0.161013 repetitions: 100
threads_per_row = 8 work_group_size = 128 
time: 0.138741 repetitions: 100
threads_per_row = 8 work_group_size = 256 
time: 0.139434 repetitions: 100
threads_per_row = 8 work_group_size = 512 
time: 0.141819 repetitions: 100
threads_per_row = 8 work_group_size = 1024
time: 0.148168 repetitions: 100
threads_per_row = 16 work_group_size = 64 
time: 0.193024 repetitions: 100
threads_per_row = 16 work_group_size = 128
time: 0.149216 repetitions: 100
threads_per_row = 16 work_group_size = 256
time: 0.147403 repetitions: 100
threads_per_row = 16 work_group_size = 512
time: 0.163763 repetitions: 100
threads_per_row = 16 work_group_size = 1024
time: 0.170920 repetitions: 100
threads_per_row = 32 work_group_size = 64
time: 0.282587 repetitions: 100
threads_per_row = 32 work_group_size = 128
time: 0.170251 repetitions: 100
threads_per_row = 32 work_group_size = 256
time: 0.174907 repetitions: 100
threads_per_row = 32 work_group_size = 512
time: 0.184007 repetitions: 100
threads_per_row = 32 work_group_size = 1024
time: 0.201341 repetitions: 100
threads_per_row = 64 work_group_size = 64
time: 0.453125 repetitions: 100
threads_per_row = 64 work_group_size = 128
time: 0.292062 repetitions: 100
threads_per_row = 64 work_group_size = 256
time: 0.301526 repetitions: 100
threads_per_row = 64 work_group_size = 512
time: 0.318690 repetitions: 100
threads_per_row = 64 work_group_size = 1024
time: 0.358805 repetitions: 100
Number of errors: 147342
*/

#define MAX_SOURCE_SIZE 16384
#define WORKGROUP_SIZE 128
#define REPEAT 100

int matrix_vector_multi(struct mtx_CSR csr_matrix, float* vector, float* output) {

    // Load kernel source file.
    FILE* file;
    file = fopen("kernel.cl", "r");
    if (!file) {
        printf("Could not load the kernel.\n");
        return 1;
    }

    char* source = (char*)malloc(MAX_SOURCE_SIZE);
    size_t source_size = fread(source, 1, MAX_SOURCE_SIZE, file);
    source[source_size] = '\0';
    fclose(file);

    // Boiler plate setup of platforms, devices, context and command queue.
    cl_int cl_status;
    cl_uint n_platforms;

    cl_status = clGetPlatformIDs(0, NULL, &n_platforms);
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * n_platforms);
    cl_status = clGetPlatformIDs(n_platforms, platforms, NULL);

    cl_uint n_devices;
    cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &n_devices);
    n_devices = 1;

    cl_device_id* devices = (cl_device_id*)malloc(sizeof(cl_device_id) * n_devices);
    cl_status = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, n_devices, devices, NULL);

    cl_context context = clCreateContext(NULL, n_devices, devices, NULL, NULL, &cl_status);
    cl_command_queue cmd_queue = clCreateCommandQueue(context, devices[0], 0, &cl_status);

    cl_program program = clCreateProgramWithSource(context, 1, (const char**)&source, NULL, &cl_status);
    cl_status = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

    // Allocate memory on device.
    cl_mem row_ptr_device = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, (csr_matrix.num_rows + 1) * sizeof(int), csr_matrix.rowptr, &cl_status);
    cl_mem col_device = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, csr_matrix.num_nonzeros * sizeof(int), csr_matrix.col, &cl_status);
    cl_mem data_device = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, csr_matrix.num_nonzeros * sizeof(float), csr_matrix.data, &cl_status);

    cl_mem vector_device = clCreateBuffer(context, CL_MEM_READ_ONLY, csr_matrix.num_cols * sizeof(float), NULL, &cl_status);
    cl_mem output_device = clCreateBuffer(context, CL_MEM_READ_WRITE, csr_matrix.num_cols * sizeof(float), NULL, &cl_status);

    // Divide work among the workgroups.
    int threads_per_row = 8;
    size_t local_item_size = WORKGROUP_SIZE;
    int n_groups = (csr_matrix.num_rows*threads_per_row - 1) / local_item_size + 1;
    size_t global_item_size = n_groups * local_item_size;

    // Create kernel and add arguments.
    cl_kernel kernel = clCreateKernel(program, "matrix_vector_multi", &cl_status);
    cl_status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&row_ptr_device);
    cl_status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&col_device);
    cl_status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&data_device);
    cl_status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&vector_device);
    cl_status |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&output_device);
    cl_status |= clSetKernelArg(kernel, 5, local_item_size*sizeof(float), NULL);
    cl_status |= clSetKernelArg(kernel, 6, sizeof(cl_int), (void*)&(csr_matrix.num_rows));
    cl_status |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void*)&threads_per_row);

    double elapsed_time = omp_get_wtime();
    for (int i = 0; i < REPEAT; i++) {
        cl_status = clEnqueueWriteBuffer(cmd_queue, vector_device, CL_TRUE, 0, csr_matrix.num_cols * sizeof(cl_float), vector, 0, NULL, NULL);
        cl_status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
        cl_status = clEnqueueReadBuffer(cmd_queue, output_device, CL_TRUE, 0, csr_matrix.num_rows * sizeof(cl_float), output, 0, NULL, NULL);
    }
    elapsed_time = omp_get_wtime() - elapsed_time;

    printf("time: %.6lf repetitions: %d\n", elapsed_time, REPEAT);

    // Close resources and free memory.
    cl_status = clFlush(cmd_queue);
    cl_status = clFinish(cmd_queue);
    cl_status = clReleaseKernel(kernel);
    cl_status = clReleaseProgram(program);
    cl_status = clReleaseMemObject(row_ptr_device);
    cl_status = clReleaseMemObject(col_device);
    cl_status = clReleaseMemObject(data_device);
    cl_status = clReleaseMemObject(vector_device);
    cl_status = clReleaseMemObject(output_device);
    cl_status = clReleaseCommandQueue(cmd_queue);
    cl_status = clReleaseContext(context);

    free(devices);
    free(platforms);
    free(source);
}

int main(int argc, char** argv) {

    if (argc < 2) {
        fprintf(stderr, "Not enough arguments\n");
        exit(1);
    }

    FILE* file = fopen(argv[1], "r");
    if (file == NULL) {
        fprintf(stderr, "Cannot open matrix file.\n");
        exit(1);
    }

    struct mtx_COO coo_matrix;
    struct mtx_CSR csr_matrix;

    int status = mtx_COO_create_from_file(&coo_matrix, file);
    if (status != 0) {
        fprintf(stderr, "Matrix file in a wrong format.\n");
        exit(1);
    }

    mtx_CSR_create_from_mtx_COO(&csr_matrix, &coo_matrix);
    mtx_COO_free(&coo_matrix);

    // Instantiate vector.
    float* vector = (float*)malloc(csr_matrix.num_cols * sizeof(float));
    for (int i = 0; i < csr_matrix.num_cols; i++) {
        vector[i] = 1.0;
    }

    float* output = (float*)malloc(csr_matrix.num_rows * sizeof(float));
    matrix_vector_multi(csr_matrix, vector, output);

    // Check the results using sequential implementation.
    float* compare_output = (float*)malloc(csr_matrix.num_rows * sizeof(float));
    for (int i = 0; i < csr_matrix.num_rows; i++) {
        compare_output[i] = 0.0;
    }

    for (int i = 0; i < csr_matrix.num_rows; i++) {
        for (int j = csr_matrix.rowptr[i]; j < csr_matrix.rowptr[i + 1]; j++) {
            compare_output[i] += csr_matrix.data[j] * vector[csr_matrix.col[j]];
        }
    }

    int n_errors = 0;
    for (int i = 0; i < csr_matrix.num_rows; i++) {
        if (fabs(output[i] - compare_output[i]) > 1e-4) {
            n_errors++;
        }
    }

    printf("Number of errors: %d\n", n_errors);

    mtx_CSR_free(&csr_matrix);
    free(vector);
    free(output);

    return 0;
}