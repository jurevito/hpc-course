#include <CL/cl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hist_equal.h"

#define BINS 256
#define MAX_SOURCE_SIZE 16384
#define WORKGROUP_SIZE 256

int hist_equal_gpu(unsigned char* image, int width, int height, int cpp) {

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

    double start_time = omp_get_wtime();
    int image_size = width * height;

    histogram hist;
    hist.R = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    hist.G = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    hist.B = (unsigned int*)calloc(BINS, sizeof(unsigned int));

    // Allocate memory on device.
    cl_mem image_device = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, image_size * cpp * sizeof(unsigned char), image, &cl_status);
    cl_mem r_device = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned int), hist.R, &cl_status);
    cl_mem g_device = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned int), hist.G, &cl_status);
    cl_mem b_device = clCreateBuffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned int), hist.B, &cl_status);

    // Divide work among the workgroups.
    size_t local_item_size = WORKGROUP_SIZE;
    size_t n_groups = ((image_size) - 1) / local_item_size + 1;
    size_t global_item_size = n_groups * local_item_size;

    // Create kernel for finding histogram of the image.
    cl_kernel kernel = clCreateKernel(program, "hist", &cl_status);
    cl_status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&image_device);
    cl_status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&r_device);
    cl_status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&g_device);
    cl_status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&b_device);
    cl_status |= clSetKernelArg(kernel, 4, BINS * sizeof(unsigned int), NULL);
    cl_status |= clSetKernelArg(kernel, 5, BINS * sizeof(unsigned int), NULL);
    cl_status |= clSetKernelArg(kernel, 6, BINS * sizeof(unsigned int), NULL);
    cl_status |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void*)&image_size);
    cl_status |= clSetKernelArg(kernel, 8, sizeof(cl_int), (void*)&cpp);

    cl_status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    // Divide work among the workgroups.
    local_item_size = WORKGROUP_SIZE;
    n_groups = 1;
    global_item_size = n_groups * local_item_size;

    // Create kernel for calculating cumulative distribution and new values.
    kernel = clCreateKernel(program, "cum_dist", &cl_status);
    cl_status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&r_device);
    cl_status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&g_device);
    cl_status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&b_device);
    cl_status |= clSetKernelArg(kernel, 3, sizeof(cl_int), (void*)&image_size);

    cl_status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    // Divide work among the workgroups.
    local_item_size = WORKGROUP_SIZE;
    n_groups = ((image_size) - 1) / local_item_size + 1;
    global_item_size = n_groups * local_item_size;

    // Create kernel for transforming the image.
    kernel = clCreateKernel(program, "transform", &cl_status);
    cl_status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&image_device);
    cl_status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&r_device);
    cl_status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&g_device);
    cl_status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&b_device);
    cl_status |= clSetKernelArg(kernel, 4, sizeof(cl_int), (void*)&image_size);
    cl_status |= clSetKernelArg(kernel, 5, sizeof(cl_int), (void*)&cpp);

    cl_status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    // Copy results from device back to host.
    cl_status = clEnqueueReadBuffer(cmd_queue, image_device, CL_TRUE, 0, image_size * cpp * sizeof(unsigned char), image, 0, NULL, NULL);

    double elapsed = omp_get_wtime() - start_time;
    printf("GPU time: %.3lf\n", elapsed);

    // Close resources and free memory.
    cl_status = clFlush(cmd_queue);
    cl_status = clFinish(cmd_queue);
    cl_status = clReleaseKernel(kernel);
    cl_status = clReleaseProgram(program);
    cl_status = clReleaseMemObject(image_device);
    cl_status = clReleaseMemObject(r_device);
    cl_status = clReleaseMemObject(g_device);
    cl_status = clReleaseMemObject(b_device);
    cl_status = clReleaseCommandQueue(cmd_queue);
    cl_status = clReleaseContext(context);

    free(devices);
    free(platforms);
    free(source);

    free(hist.R);
    free(hist.G);
    free(hist.B);

    return 0;
}
