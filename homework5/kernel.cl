__kernel void rgb(__global const unsigned char* image,
    			  __global unsigned int* r,
    			  __global unsigned int* g,
    			  __global unsigned int* b,
    			  __local unsigned int* r_local,
    			  __local unsigned int* g_local,
    			  __local unsigned int* b_local,
    			  int size,
    			  int hist_size,
    			  int cpp) {

    int lid = get_local_id(0);
    int gid = get_global_id(0);

    if (lid == 0) {
        for (int i = 0; i < hist_size; i++) {
            r_local[i] = 0;
            g_local[i] = 0;
            b_local[i] = 0;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&r_local[image[gid * cpp + 0]]);
    atomic_inc(&g_local[image[gid * cpp + 1]]);
    atomic_inc(&b_local[image[gid * cpp + 2]]);
    // r_local[image[gid*cpp + 0]] += 1;
    // g_local[image[gid*cpp + 1]] += 1;
    // b_local[image[gid*cpp + 2]] += 1;

    barrier(CLK_LOCAL_MEM_FENCE);

    // atomic_add(&r[image[gid*cpp + 0]], 1);
    // atomic_add(&g[image[gid*cpp + 1]], 1);
    // atomic_add(&b[image[gid*cpp + 2]], 1);

    // while (gid < size) {
    //	r_local[image[gid*cpp + 0]] += 1;
    //	g_local[image[gid*cpp + 1]] += 1;
    //	b_local[image[gid*cpp + 2]] += 1;
    //	gid += get_global_size(0);
    // }
    atomic_add(&r[lid], r_local[lid]);
    atomic_add(&g[lid], g_local[lid]);
    atomic_add(&b[lid], b_local[lid]);
    /*
    if (lid == 0) {
            for (int i = 0 ; i<256 ; i++) {
                    atomic_add(&r[i], r_local[i]);
                    atomic_add(&g[i], g_local[i]);
                    atomic_add(&b[i], b_local[i]);
                    //r[i] += r_local[i];
                    //g[i] += g_local[i];
                    //b[i] += b_local[i];
            }
    }
    */
}
