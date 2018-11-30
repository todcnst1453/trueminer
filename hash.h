#ifndef  _HASH_H_
#define  _HASH_H_

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>

#define		DATA_LENGTH			2048
#define		PMT_SIZE			4
#define     TBL_SIZE			16
#define		HEAD_SIZE			32
#define		DGST_SIZE			32
#define		TARG_SIZE			16
typedef		unsigned int		uint32;
typedef		unsigned long long 	uint64;
typedef		unsigned char		byte;


__device__ void fchainhash(uint64 nonce, byte digs[DGST_SIZE]);
__global__ void compute(uint64 nonce_start);

#endif
