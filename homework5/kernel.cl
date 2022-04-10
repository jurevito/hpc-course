__kernel void rgb(__global const unsigned char* image,
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

	while (gid < size/cpp) {
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
