__kernel void matrix_vector_multi(__global const int* rowptr,
                                  __global const int* col,
                                  __global const float* data,
                                  __global const float* vector,
                                  __global float* output,
                                  __local float* partial,
                                  const int rows,
                                  const int threads_per_row) {
    int gid = get_global_id(0);
    int lid = get_local_id(0);
    
    int row_id = gid / threads_per_row;
    int lid_by_row = lid % threads_per_row;

    if(row_id < rows) {
        float partial_sum = 0.0f;
        for(int i = rowptr[row_id] + lid_by_row ; i<rowptr[row_id+1] ; i+=threads_per_row) {
            partial_sum += data[i] * vector[col[i]];
        }

        partial[lid] = partial_sum;

        barrier(CLK_LOCAL_MEM_FENCE);

        // Row reduction.
        if(lid_by_row == 0) {
            float sum = 0.0f;
            for(int i = lid; i<lid+threads_per_row; i++) {
                sum += partial[i];
            }

            output[row_id] = sum;
        }
    }

    /*
    barrier(CLK_LOCAL_MEM_FENCE);

	int floorPow2 = exp2(log2((float)get_local_size(0)));
    if (get_local_size(0) != floorPow2)										
	{
		if ( lid >= floorPow2 )
            partial[lid - floorPow2] += partial[lid];
		barrier(CLK_LOCAL_MEM_FENCE);
    }

	for(int i = (floorPow2>>1); i>0; i >>= 1) 
	{
		if(lid < i) 
			partial[lid] += partial[lid + i];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
    */

    /*
    if(gid < rows) {
        float sum = 0.0f;
        for (int j = rowptr[gid]; j < rowptr[gid + 1]; j++) {
            sum += data[j] * vector[col[j]];
        }

        output[gid] = sum;
    }
    */
}