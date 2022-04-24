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
    int local_offset = (lid / threads_per_row)*threads_per_row;

    if(row_id < rows) {
        float partial_sum = 0.0f;
        for(int i = rowptr[row_id] + lid_by_row ; i<rowptr[row_id+1] ; i+=threads_per_row) {
            partial_sum += data[i] * vector[col[i]];
        }

        partial[lid] = partial_sum;

        barrier(CLK_LOCAL_MEM_FENCE);

        int floor_pow2 = exp2(log2((float)threads_per_row));
        if (threads_per_row != floor_pow2) {
            if(floor_pow2 <= lid_by_row) {
                partial[local_offset + lid_by_row - floor_pow2] += partial[local_offset + lid_by_row];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        for(int i = floor_pow2>>1 ; i>0 ; i>>=1) {
            if(lid_by_row < i) {
                partial[local_offset + lid_by_row] += partial[local_offset + lid_by_row + i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if(lid_by_row == 0) {
            output[row_id] = partial[local_offset];
        }
    }
}