__kernel void hist(__global const unsigned char* image,
                   __global unsigned int* r,
                   __global unsigned int* g,
                   __global unsigned int* b,
                   __local unsigned int* r_local,
                   __local unsigned int* g_local,
                   __local unsigned int* b_local,
                   int size,
                   int cpp) {

    int lid = get_local_id(0);
    int gid = get_global_id(0);

    r_local[lid] = 0;
    g_local[lid] = 0;
    b_local[lid] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    while (gid < size) {
        atomic_inc(&r_local[image[gid * cpp + 0]]);
        atomic_inc(&g_local[image[gid * cpp + 1]]);
        atomic_inc(&b_local[image[gid * cpp + 2]]);
        gid += get_global_size(0);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    
    atomic_add(&r[lid], r_local[lid]);
    atomic_add(&g[lid], g_local[lid]);
    atomic_add(&b[lid], b_local[lid]);
}

__kernel void cum_dist(__global unsigned int* r,
                       __global unsigned int* g,
                       __global unsigned int* b,
                       __global unsigned int* r_dist,
                       __global unsigned int* g_dist,
                       __global unsigned int* b_dist,
                       const int size) {
    
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    for(int i = 2 ; i<=256 ; i*=2) {
        int index = (i-1)+(lid*i);

        if(index < 256) {
            r_dist[index] = r_dist[index] + r_dist[index - (i/2)];
            g_dist[index] = g_dist[index] + g_dist[index - (i/2)];
            b_dist[index] = b_dist[index] + b_dist[index - (i/2)];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for(int i = 128 ; i>=2 ; i/=2) {
        int index = (i-1)+(lid*i);

        if(index + (i/2) < 256) {
            r_dist[index + (i/2)] = r_dist[index + (i/2)] + r_dist[index];
            g_dist[index + (i/2)] = g_dist[index + (i/2)] + g_dist[index];
            b_dist[index + (i/2)] = b_dist[index + (i/2)] + b_dist[index];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    r[lid] = (unsigned int)round((float)(r_dist[lid] - r_dist[0]) / (float)(size - r_dist[0]) * (256 - 1));
    g[lid] = (unsigned int)round((float)(g_dist[lid] - g_dist[0]) / (float)(size - g_dist[0]) * (256 - 1));
    b[lid] = (unsigned int)round((float)(b_dist[lid] - b_dist[0]) / (float)(size - b_dist[0]) * (256 - 1));
}

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
}