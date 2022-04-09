#include <CL/cl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
//#include "stb_image_write.h"

#define BINS 256

typedef struct _histogram {
    unsigned int* R;
    unsigned int* G;
    unsigned int* B;
} histogram;

void hist_cpu(unsigned char* imageIn, histogram H, int width, int height, int cpp) {
    // Each color channel is 1 byte long, there are 4 channels RED, BLUE, GREEN,  and ALPHA
    // The order is RED|GREEN|BLUE|ALPHA for each pixel, we ignore the ALPHA channel when computing the histograms
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            H.R[imageIn[(i * width + j) * cpp]]++;
            H.G[imageIn[(i * width + j) * cpp + 1]]++;
            H.B[imageIn[(i * width + j) * cpp + 2]]++;
        }
    }
}

void hist_gpu(unsigned char* imageIn, histogram H, int width, int height, int cpp) {
}

void print_hist(histogram H) {
    printf("Colour\tNo. Pixels\n");
    for (int i = 0; i < BINS; i++) {
        if (H.R[i] > 0)
            printf("%dR\t%d\n", i, H.R[i]);
        if (H.G[i] > 0)
            printf("%dG\t%d\n", i, H.G[i]);
        if (H.B[i] > 0)
            printf("%dB\t%d\n", i, H.B[i]);
    }
}
int main(int argc, char** argv) {

    if (argc < 2) {
        fprintf(stderr, "Not enough arguments\n");
        exit(1);
    }

    char* img_file = argv[1];

    // Initalize the histogram
    histogram hist;
    hist.B = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    hist.G = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    hist.R = (unsigned int*)calloc(BINS, sizeof(unsigned int));

    int width, height, cpp;
    unsigned char* image_in = stbi_load(img_file, &width, &height, &cpp, 0);

    if (image_in) {
        double start_time = omp_get_wtime();
        hist_cpu(image_in, hist, width, height, cpp);
        double elapsed = omp_get_wtime() - start_time;
        printf("CPU time: %.3lf\n", elapsed);

        start_time = omp_get_wtime();
        hist_gpu(image_in, hist, width, height, cpp);
        elapsed = omp_get_wtime() - start_time;
        printf("GPU time: %.3lf\n", elapsed);
    } else {
        fprintf(stderr, "Error loading image %s!\n", img_file);
        exit(1);
    }

    return 0;
}