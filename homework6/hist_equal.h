#ifndef HIST_H
#define HIST_H

typedef struct _histogram {
    unsigned int* R;
    unsigned int* G;
    unsigned int* B;
} histogram;

int hist_equal_gpu(unsigned char* image, int width, int height, int cpp);

#endif