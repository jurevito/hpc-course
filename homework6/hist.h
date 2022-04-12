#ifndef HIST_H
#define HIST_H

typedef struct _histogram {
    unsigned int* R;
    unsigned int* G;
    unsigned int* B;
} histogram;

void hist_cpu(unsigned char* image, histogram hist, int width, int height, int cpp);
void hist_gpu(unsigned char* image, histogram hist, int width, int height, int cpp);

#endif