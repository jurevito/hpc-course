__kernel void transform(__global unsigned char* image,
                        __global const unsigned int* r,
                        __global const unsigned int* g,
                        __global const unsigned int* b,
                        const int size,
                        const int cpp) {

    int lid = get_local_id(0);
    int gid = get_global_id(0);

    while (gid < size) {
        image[gid * cpp + 0] = r[image[gid * cpp + 0]];
        image[gid * cpp + 1] = g[image[gid * cpp + 1]];
        image[gid * cpp + 2] = b[image[gid * cpp + 2]];
        gid += get_global_size(0);
    }
    /*
    for(int i = 0 ; i<(height*width*cpp) ; i+=cpp) {
        image[i+0] = hist.R[image[i+0]];
        image[i+1] = hist.G[image[i+1]];
        image[i+2] = hist.B[image[i+2]];
    }
    */
}
