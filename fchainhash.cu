#include "gpu_var.h"
#include "hash.h"
#include "sha3.h"

__device__ uint64 gFoundIdx = uint64(-1);
__constant__ int kXor[256];

int genLookupTable(uint64 *plookup, uint32 *ptable)
{
	uint64 *ptbl = plookup;
	int lkt_wz = (DATA_LENGTH) / 64;
	int lkt_sz = DATA_LENGTH*lkt_wz;
	
	int idx = 0;
	for (int k = 0; k < 16; k++)
	{
		uint64 *plkt = plookup+k*lkt_sz;
		uint32 *ptbl = ptable + k*DATA_LENGTH*PMT_SIZE;
		for (int x = 0; x < DATA_LENGTH; x++)
		{
			if (x == 0 && k == 13)
				x = x;
			for (int y = 0; y < PMT_SIZE; y++)
			{
				int val = *ptable;
				if (val == 0xFFF)
				{
					ptable++;
					continue;
				}
					
				int v_i = val / 64;
				int v_r = val % 64;
				plkt[v_i] |= ((uint64)1 << v_r);
				ptable++;
			}		
			plkt += lkt_wz;
		}
		//printf("\n");
	}

	return 0;
}

__device__ int xor64(uint64 val)
{
	int r = 0;

	for (int k = 0; k < 8 && val; k++)
	{
		r ^= kXor[val & 0xFF];
		val >>= 8;
	}
	return r;
}

__device__ int muliple(uint64 input[32], uint64 *prow)
{
	int r = 0;
	for (int k = 0; k < 32; k++)
	{
		if (input[k] != 0 && prow[k] != 0)
			r ^= xor64(input[k] & prow[k]);
	}

	return r;
}

__device__ int MatMuliple(uint64 input[32], uint64 output[32], uint64 *pmat)
{
	uint64 *prow = pmat;

	for (int k = 0; k < 2048; k++)
	{
		int k_i = k / 64;
		int k_r = k % 64;
		unsigned int temp;
		temp = muliple(input, prow);

		output[k_i] |= ((uint64)temp << k_r);
		prow += 32;
	}

	return 0;
}

__device__ int shift2048(uint64 in[32], int sf)
{
	int sf_i = sf / 64;
	int sf_r = sf % 64;
	uint64 mask = ((uint64)1 << sf_r) - 1;
	int bits = (64 - sf_r);
	uint64 res;
	if (sf_i == 1)
	{
		uint64 val = in[0];
		for (int k = 0; k < 31; k++)
		{
			in[k] = in[k + 1];
		}
		in[31] = val;
	}
	res = (in[0] & mask) << bits;
	for (int k = 0; k < 31; k++)
	{
		uint64 val = (in[k + 1] & mask) << bits;
		in[k] = (in[k] >> sf_r) + val;
	}
	in[31] = (in[31] >> sf_r) + res;
	return 0;
}

__device__ int scramble(uint64 *permute_in)
{
	uint64 *ptbl;
	uint64 permute_out[32] = { 0 };
	for (int k = 0; k < 64; k++)
	{
		int sf, bs;
		sf = permute_in[0] & 0x7f;
		bs = permute_in[31] >> 60;
		ptbl = gTable + bs * 2048 * 32;
		MatMuliple(permute_in, permute_out, ptbl);

		shift2048(permute_out, sf);
		for (int k = 0; k < 32; k++)
		{
			permute_in[k] = permute_out[k];
			permute_out[k] = 0;
		}	
	}

	return 0;
}

__device__ int byteReverse(byte sha512_out[64])
{
	for (int k = 0; k < 32; k++)
	{
		byte temp = sha512_out[k];
		sha512_out[k] = sha512_out[63 - k];
		sha512_out[63 - k] = temp;
	}

	return 0;
}

int convertLE(byte header[HEAD_SIZE])
{
	int wz = HEAD_SIZE / 4;

	for (int k = 0; k < wz; k++)
	{
		byte temp[4];
		temp[0] = header[k * 4 + 3];
		temp[1] = header[k * 4 + 2];
		temp[2] = header[k * 4 + 1];
		temp[3] = header[k * 4 + 0];
		header[k * 4 + 0] = temp[0];
		header[k * 4 + 1] = temp[1];
		header[k * 4 + 2] = temp[2];
		header[k * 4 + 3] = temp[3];
	}
	return 0;
}

__device__ int convertWD(byte header[HEAD_SIZE])
{
	byte temp[HEAD_SIZE];
	int wz = HEAD_SIZE / 4;
	for (int k = 0; k < wz; k++)
	{
		int i = 7 - k;
		temp[k * 4] = header[i * 4];
		temp[k * 4 + 1] = header[i * 4 + 1];
		temp[k * 4 + 2] = header[i * 4 + 2];
		temp[k * 4 + 3] = header[i * 4 + 3];
	}
	for (int k = 0; k < HEAD_SIZE; k++)
	{
		header[k] = temp[k];
	}
	return 0;
}

__device__ int compare(byte dgst[DGST_SIZE], byte target1[TARG_SIZE], byte target2[TARG_SIZE])
{
	for (int k = TARG_SIZE - 1; k >= 0; k--)
	{
		int dif = (int)dgst[k] - (int)target1[k];
		if (dif > 0)
			return 0;
		if (dif < 0)
			return 1;
	}
	for (int k = TARG_SIZE - 1; k >= 0; k--)
	{
		int dif = (int)dgst[k + 16] - (int)target2[k];
		if (dif > 0)
			return 0;
		if (dif < 0)
			return 1;
	}
	return 0;
}


__global__ void compute(uint64 nonce_start)
{
	byte digs[DGST_SIZE];
	const uint64 offset = gridDim.x * blockDim.x;
	nonce_start += threadIdx.x + blockIdx.x * blockDim.x;

	while (nonce_start < gFoundIdx)
	{
		fchainhash(nonce_start, digs);

		if (compare(digs, kTarget1, kTarget2) == 1)
		{
			atomicMin((unsigned long long int*)&gFoundIdx, unsigned long long int(nonce_start));
			break;
		}
		// Get result here
		printf("Current nonce : %llu\n", nonce_start);
		nonce_start += offset;
	}
}

__device__ void fchainhash(uint64 nonce, byte digs[DGST_SIZE])
{
	byte seed[64] = { 0 };
	byte output[DGST_SIZE] = { 0 };

	uint32 val0 = (uint32)(nonce & 0xFFFFFFFF);
	uint32 val1 = (uint32)(nonce >> 32);
	for (int k = 3; k >= 0; k--)
	{
		seed[k] = val0 & 0xFF;
		val0 >>= 8;
	}
	
	for (int k = 7; k >= 4; k--)
	{
		seed[k] = val1 & 0xFF;
		val1 >>= 8;
	}
	
	for (int k = 0; k < HEAD_SIZE; k++)
	{
		seed[k+8] = kInput[k];
	}

	byte sha512_out[64];
	sha3(seed, 64, sha512_out, 64);
	byteReverse(sha512_out);
	uint64 permute_in[32] = { 0 }, permute_out[32] = {0};
	for (int k = 0; k < 8; k++)
	{
		for (int x = 0; x < 8; x++)
		{
			int sft = x * 8;
			uint64 val = ((uint64)sha512_out[k*8+x] << sft);
			permute_in[k] += val;
		}		
	}

	for (int k = 1; k < 4; k++)
	{
		for (int x = 0; x < 8; x++)
			permute_in[k * 8 + x] = permute_in[x];
	}

	scramble(permute_in);
	
	byte dat_in[256];
	for (int k = 0; k < 32; k++)
	{
		uint64 val = permute_in[k];
		for (int x = 0; x < 8; x++)
		{
			dat_in[k * 8 + x] = val & 0xFF;
			val = val >> 8;
		}
	}
	
	for (int k = 0; k < 64; k++)
	{
		byte temp;
		temp = dat_in[k * 4];
		dat_in[k * 4] = dat_in[k * 4 + 3];
		dat_in[k * 4 + 3] = temp;
		temp = dat_in[k * 4 + 1];
		dat_in[k * 4 + 1] = dat_in[k * 4 + 2];
		dat_in[k * 4 + 2] = temp;
	}

	//unsigned char output[64];
	sha3(dat_in, 256, output, 32);
	// reverse byte
	for (int k = 0; k < DGST_SIZE; k++)
	{
		digs[k] = output[DGST_SIZE - k - 1];
	}
}

