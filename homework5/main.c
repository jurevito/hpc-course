#include <CL/cl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"

#define BINS 256
#define MAX_SOURCE_SIZE 16384
#define WORKGROUP_SIZE 256

typedef struct _histogram {
    unsigned int* R;
    unsigned int* G;
    unsigned int* B;
} histogram;

void hist_cpu(unsigned char* image, histogram hist, int width, int height, int cpp) {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            hist.R[image[(i * width + j) * cpp]]++;
            hist.G[image[(i * width + j) * cpp + 1]]++;
            hist.B[image[(i * width + j) * cpp + 2]]++;
        }
    }
}

void hist_gpu(unsigned char* image, histogram hist, int width, int height, int cpp) {

    // Load kernel source file.
    FILE* file;
    file = fopen("kernel.cl", "r");
    if (!file) {
        printf("Could not load the kernel.\n");
        return;
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

    int image_size = width * height * cpp;
    int hist_size = BINS;

    // Divide work among the workgroups.
    size_t local_item_size = WORKGROUP_SIZE;
    size_t n_groups = ((image_size / cpp) - 1) / local_item_size + 1;
    size_t global_item_size = n_groups * local_item_size;

    // Allocate memory on device.
    cl_mem image_device = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, image_size * sizeof(unsigned char), image, &cl_status);
    cl_mem r_device = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, hist_size * sizeof(unsigned int), hist.R, &cl_status);
    cl_mem g_device = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, hist_size * sizeof(unsigned int), hist.G, &cl_status);
    cl_mem b_device = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, hist_size * sizeof(unsigned int), hist.B, &cl_status);

    // Create kernel and add arguments.
    cl_kernel kernel = clCreateKernel(program, "rgb", &cl_status);
    cl_status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&image_device);
    cl_status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&r_device);
    cl_status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&g_device);
    cl_status |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&b_device);
    cl_status |= clSetKernelArg(kernel, 4, hist_size * sizeof(unsigned int), NULL);
    cl_status |= clSetKernelArg(kernel, 5, hist_size * sizeof(unsigned int), NULL);
    cl_status |= clSetKernelArg(kernel, 6, hist_size * sizeof(unsigned int), NULL);
    cl_status |= clSetKernelArg(kernel, 7, sizeof(cl_int), (void*)&image_size);
    cl_status |= clSetKernelArg(kernel, 8, sizeof(cl_int), (void*)&hist_size);
    cl_status |= clSetKernelArg(kernel, 9, sizeof(cl_int), (void*)&cpp);

    cl_status = clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    printf("5. cl_status = %d\n", cl_status);

    // Copy results from device back to host.
    cl_status = clEnqueueReadBuffer(cmd_queue, r_device, CL_TRUE, 0, hist_size * sizeof(unsigned int), hist.R, 0, NULL, NULL);
    cl_status = clEnqueueReadBuffer(cmd_queue, g_device, CL_TRUE, 0, hist_size * sizeof(unsigned int), hist.G, 0, NULL, NULL);
    cl_status = clEnqueueReadBuffer(cmd_queue, b_device, CL_TRUE, 0, hist_size * sizeof(unsigned int), hist.B, 0, NULL, NULL);

    printf("6. cl_status = %d\n", cl_status);

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
}

int main(int argc, char** argv) {

    if (argc < 2) {
        fprintf(stderr, "Not enough arguments\n");
        exit(1);
    }

    char* img_file = argv[1];

    int width, height, cpp;
    unsigned char* image_in = stbi_load(img_file, &width, &height, &cpp, 0);

    if (image_in) {
        histogram hist;
        hist.R = (unsigned int*)calloc(BINS, sizeof(unsigned int));
        hist.G = (unsigned int*)calloc(BINS, sizeof(unsigned int));
        hist.B = (unsigned int*)calloc(BINS, sizeof(unsigned int));

        double start_time = omp_get_wtime();
        hist_cpu(image_in, hist, width, height, cpp);
        double elapsed = omp_get_wtime() - start_time;
        printf("CPU time: %.3lf results: (%d,%d,%d)\n", elapsed, hist.R[50], hist.G[50], hist.B[50]);

        free(hist.R);
        free(hist.G);
        free(hist.B);

        hist.R = (unsigned int*)calloc(BINS, sizeof(unsigned int));
        hist.G = (unsigned int*)calloc(BINS, sizeof(unsigned int));
        hist.B = (unsigned int*)calloc(BINS, sizeof(unsigned int));

        start_time = omp_get_wtime();
        hist_gpu(image_in, hist, width, height, cpp);
        elapsed = omp_get_wtime() - start_time;
        printf("GPU time: %.3lf results: (%d,%d,%d)\n", elapsed, hist.R[50], hist.G[50], hist.B[50]);

        free(hist.R);
        free(hist.G);
        free(hist.B);
    } else {
        fprintf(stderr, "Error loading image %s!\n", img_file);
        exit(1);
    }

    return 0;
}