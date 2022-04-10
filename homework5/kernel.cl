__kernel void rgb(	__global const unsigned char* image, 
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

	while (gid < size) {
		r_local[image[gid*cpp + 0]] += 1;
		g_local[image[gid*cpp + 1]] += 1;
		b_local[image[gid*cpp + 2]] += 1;
		gid += get_global_size(0);
	}

	if (lid == 0) {
		for (int i = 0 ; i<256 ; i++) {
			//atomic_add(&r[i], r_local[i]);
			//atomic_add(&g[i], g_local[i]);
			//atomic_add(&b[i], b_local[i]);
			r[i] += r_local[i];
			g[i] += g_local[i];
			b[i] += b_local[i];
		}
	}
}

/*__kernel void dot_product(__global const float *a,
						 __global const float *b,		
						 __global float *c,		
						 int size,
						 __local float *partial) {

	int lid  = get_local_id(0);
    int gid = get_global_id(0); 
 
	float sum = 0.0f;
	while (gid < size) {
		sum += a[gid] * b[gid];
		gid += get_global_size(0);
	}

	partial[lid] = sum;
	barrier(CLK_LOCAL_MEM_FENCE);

	if(lid == 0) {
		sum = 0.0f;
		for(int i = 0; i < get_local_size(0); i++) {
			sum += partial[i];
		}

		c[get_group_id(0)] = sum;
	}
}
*/