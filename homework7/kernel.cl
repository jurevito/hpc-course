__kernel void matrix_vector_multi(__global const int* rowptr,
                       __global const int* col,
                       __global const float* data,
                       __global float* vector,
                       __global float* output,
                       int rows) {
    int gid = get_global_id(0);

    if(gid < rows) {
        float sum = 0.0f;
        for (int j = rowptr[gid]; j < rowptr[gid + 1]; j++) {
            sum += data[j] * vector[col[j]];
        }

        output[gid] = sum;
    }
}