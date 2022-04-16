#include <CL/cl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "hist.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BINS 256
#define MAX_SOURCE_SIZE 16384
#define WORKGROUP_SIZE 256

int cumulative_dist_gpu(histogram cum_dist, histogram hist) {

    // Load kernel source file.
    FILE* file;
    file = fopen("scan_kernel.cl", "r");
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

    // Divide work among the workgroups.
    size_t local_item_size = WORKGROUP_SIZE;
    size_t n_groups = (BINS - 1) / local_item_size + 1;
    size_t global_item_size = n_groups * local_item_size;

    // Allocate memory on device.
    cl_mem r_device = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned int), hist.R, &cl_status);
    cl_mem g_device = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned int), hist.G, &cl_status);
    cl_mem b_device = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, BINS * sizeof(unsigned int), hist.B, &cl_status);

    // Create kernel and add arguments.
    cl_kernel kernel = clCreateKernel(program, "cum_dist", &cl_status);
    cl_status |= clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&r_device);
    cl_status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&g_device);
    cl_status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&b_device);

    cl_status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);

    // Copy results from device back to host.
    cl_status = clEnqueueReadBuffer(cmd_queue, r_device, CL_TRUE, 0, BINS * sizeof(unsigned int), cum_dist.R, 0, NULL, NULL);
    cl_status = clEnqueueReadBuffer(cmd_queue, g_device, CL_TRUE, 0, BINS * sizeof(unsigned int), cum_dist.G, 0, NULL, NULL);
    cl_status = clEnqueueReadBuffer(cmd_queue, b_device, CL_TRUE, 0, BINS * sizeof(unsigned int), cum_dist.B, 0, NULL, NULL);

    // Close resources and free memory.
    cl_status = clFlush(cmd_queue);
    cl_status = clFinish(cmd_queue);
    cl_status = clReleaseKernel(kernel);
    cl_status = clReleaseProgram(program);
    cl_status = clReleaseMemObject(r_device);
    cl_status = clReleaseMemObject(g_device);
    cl_status = clReleaseMemObject(b_device);
    cl_status = clReleaseCommandQueue(cmd_queue);
    cl_status = clReleaseContext(context);

    free(devices);
    free(platforms);
    free(source);

    return 0;
}

int hist_equal_cpu(unsigned char* image, int width, int height, int cpp) {

    // Create the histogram of the input image.
    histogram hist;
    hist.R = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    hist.G = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    hist.B = (unsigned int*)calloc(BINS, sizeof(unsigned int));

    hist_cpu(image, hist, width, height, cpp);

    // Calculate cumulative distribution.
    histogram cum_dist; // Cumulative distribution.
    cum_dist.R = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    cum_dist.G = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    cum_dist.B = (unsigned int*)calloc(BINS, sizeof(unsigned int));

    cum_dist.R[0] = hist.R[0];
    cum_dist.G[0] = hist.G[0];
    cum_dist.B[0] = hist.B[0];

    for(int i = 1 ; i<BINS ; i++) {
        cum_dist.R[i] = hist.R[i] + cum_dist.R[i-1];
        cum_dist.G[i] = hist.G[i] + cum_dist.G[i-1];
        cum_dist.B[i] = hist.B[i] + cum_dist.B[i-1];
    }

    // Calculate new color level values.
    for(int i = 0 ; i<BINS ; i++) {
        hist.R[i] = (unsigned int)round((double)(cum_dist.R[i] - cum_dist.R[0]) / (double)(width*height - cum_dist.R[0]) * (BINS - 1));
        hist.G[i] = (unsigned int)round((double)(cum_dist.G[i] - cum_dist.G[0]) / (double)(width*height - cum_dist.G[0]) * (BINS - 1));
        hist.B[i] = (unsigned int)round((double)(cum_dist.B[i] - cum_dist.B[0]) / (double)(width*height - cum_dist.B[0]) * (BINS - 1));
    }

    // Assign the new transformed value to each color channel of a pixel.
    for(int i = 0 ; i<(height*width*cpp) ; i+=cpp) {
        image[i+0] = hist.R[image[i+0]];
        image[i+1] = hist.G[image[i+1]];
        image[i+2] = hist.B[image[i+2]];
    }

    free(hist.R);
    free(hist.G);
    free(hist.B);

    free(cum_dist.R);
    free(cum_dist.G);
    free(cum_dist.B);

    return 0;
}

int hist_equal_gpu(unsigned char* image, int width, int height, int cpp) {

    // Create the histogram of the input image.
    histogram hist;
    hist.R = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    hist.G = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    hist.B = (unsigned int*)calloc(BINS, sizeof(unsigned int));

    hist_gpu(image, hist, width, height, cpp);

    // Calculate cumulative distribution.
    histogram cum_dist; // Cumulative distribution.
    cum_dist.R = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    cum_dist.G = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    cum_dist.B = (unsigned int*)calloc(BINS, sizeof(unsigned int));

    cumulative_dist_gpu(cum_dist, hist);

    // Calculate new color level values.

    // Assign the new transformed value to each color channel of a pixel.


    return 0;
}

int main(int argc, char** argv) {

    if (argc < 2) {
        fprintf(stderr, "Not enough arguments\n");
        exit(1);
    }

    char* img_file = argv[1];

    int width, height, cpp;
    unsigned char* image = stbi_load(img_file, &width, &height, &cpp, 0);

    double start_time = omp_get_wtime();
    hist_equal_gpu(image, width, height, cpp);
    double elapsed = omp_get_wtime() - start_time;
    printf("CPU time: %.3lf\n", elapsed);

    // Save image after histogram equalization.
    char* output_name = malloc(4 + strlen(img_file) + 1);
    strcpy(output_name, "out_");
    strcat(output_name, img_file);
    stbi_write_jpg(output_name, width, height, cpp, image, width*cpp);

    return 0;
}