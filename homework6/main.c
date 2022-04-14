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

void hist_equal_cpu(unsigned char* image, int width, int height, int cpp) {

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
}

int main(int argc, char** argv) {

    if (argc < 2) {
        fprintf(stderr, "Not enough arguments\n");
        exit(1);
    }

    char* img_file = argv[1];

    int width, height, cpp;
    unsigned char* image = stbi_load(img_file, &width, &height, &cpp, 0);

    hist_equal_cpu(image, width, height, cpp);

    // Save image after histogram equalization.
    char* output_name = malloc(4 + strlen(img_file) + 1);
    strcpy(output_name, "out_");
    strcat(output_name, img_file);
    stbi_write_jpg(output_name, width, height, cpp, image, width*cpp);

    return 0;
}