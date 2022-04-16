__kernel void cum_dist(__global unsigned int* r,
                       __global unsigned int* g,
                       __global unsigned int* b
                       ) {
    
    int gid = get_global_id(0);
    int lid = get_local_id(0);

    for(int i = 2 ; i<=256 ; i*=2) {
        int index = (i-1)+(lid*i);

        if(index < 256) {
            r[index] = r[index] + r[index - (i/2)];
            g[index] = g[index] + g[index - (i/2)];
            b[index] = b[index] + b[index - (i/2)];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // second part
    
    for(int i = 128 ; i>=2 ; i/=2) {
        int index = (i-1)+(lid*i);

        if(index + (i/2) < 256) {
            r[index + (i/2)] = r[index + (i/2)] + r[index];
            g[index + (i/2)] = g[index + (i/2)] + g[index];
            b[index + (i/2)] = b[index + (i/2)] + b[index];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
}

/*
FIRST PART OF ALGORITHM
(i-1)+(lid*i)

## i = 2 ##
r[1] = r[1] + r[0] lid=0 | (2-1)+(0*2) = 1
r[3] = r[3] + r[2] lid=1 | (2-1)+(1*2) = 3
...
r[255] = r[255] + r[254] lid=127 | (2-1)+(127*2) = 255

## i = 4 ##
r[3] = r[3] + r[1] lid=0 | (4-1)+(0*4) = 3
r[7] = r[7] + r[5] lid=1 | (4-1)+(1*4) = 7
...
r[255] = r[255] + r[253] lid=63 | (4-1)+(63*4) = 255
.
.
.
## i = 256 ##
r[255] = r[255] + r[127] lid=0 | (256-1)+(0*256) = 255
*/

/*
SECOND PART OF ALGORITHM
BINS = 256
((BINS - 1) + r[i-1]) / 2
r[i-1 + (i/2)] = r[i-1 + (i/2)] + r[i-1]

## i = 128 ##
r[191] = r[191] + r[127]

r[(i-1)+(lid*i)+(i/2)] = r[(i-1)+(lid*i)+(i/2)] + r[(i-1)+(lid*i)]
## i = 64 ##
r[95] = r[95] + r[63] 
r[159] = r[159] + r[127]
r[223] = r[223] + r[191]
.
.
.
r[(i-1)+(lid*i)+(i/2)] = r[(i-1)+(lid*i)+(i/2)] + r[(i-1)+(lid*i)]
## i = 2 ##
r[2] = r[2] + r[1]
r[4] = r[4] + r[3]
r[6] = r[6] + r[5]
*/