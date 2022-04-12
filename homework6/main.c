#include <CL/cl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "hist.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define BINS 256

int main(int argc, char** argv) {

    if (argc < 2) {
        fprintf(stderr, "Not enough arguments\n");
        exit(1);
    }

    char* img_file = argv[1];

    int width, height, cpp;
    unsigned char* image_in = stbi_load(img_file, &width, &height, &cpp, 0);

    histogram hist;
    hist.R = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    hist.G = (unsigned int*)calloc(BINS, sizeof(unsigned int));
    hist.B = (unsigned int*)calloc(BINS, sizeof(unsigned int));

    hist_gpu(image_in, hist, width, height, cpp);

    free(hist.R);
    free(hist.G);
    free(hist.B);
    return 0;
}