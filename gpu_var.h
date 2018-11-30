#include "hash.h"
extern __device__ uint64 gTable[1048576];
extern __constant__ byte kInput[HEAD_SIZE];
extern __constant__ byte kTarget1[TARG_SIZE];
extern __constant__ byte kTarget2[TARG_SIZE];
extern __device__ byte gOutput[DGST_SIZE];
extern __device__ uint64 gFoundIdx;
extern __constant__ int kXor[256];