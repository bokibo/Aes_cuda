
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include "consts.h"
#include <malloc.h>	
#include <fstream>
#include <stdlib.h>
#include <iostream>
#include <time.h>

using namespace std;
//macros multiplication for Inverse MixColumns
#define xtime(x)   ((x<<1) ^ (((x>>7) & 1) * 0x1b))
#define Multiply(x,y) (((y & 1) * x) ^ ((y>>1 & 1) * xtime(x)) ^ ((y>>2 & 1) * xtime(xtime(x))) ^ ((y>>3 & 1) * xtime(xtime(xtime(x)))) ^ ((y>>4 & 1) * xtime(xtime(xtime(xtime(x))))))

__constant__ unsigned char Sbox_dev[256] =
{
	0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76 ,
	0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0 ,
	0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15 ,
	0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75 ,
	0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84 ,
	0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF ,
	0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8 ,
	0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2 ,
	0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73 ,
	0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB ,
	0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79 ,
	0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08 ,
	0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A ,
	0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E ,
	0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF ,
	0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
};

__constant__ unsigned char InvSbox_dev[256] =
{
	0x52, 0x09, 0x6A, 0xD5, 0x30, 0x36, 0xA5, 0x38, 0xBF, 0x40, 0xA3, 0x9E, 0x81, 0xF3, 0xD7, 0xFB ,
	0x7C, 0xE3, 0x39, 0x82, 0x9B, 0x2F, 0xFF, 0x87, 0x34, 0x8E, 0x43, 0x44, 0xC4, 0xDE, 0xE9, 0xCB ,
	0x54, 0x7B, 0x94, 0x32, 0xA6, 0xC2, 0x23, 0x3D, 0xEE, 0x4C, 0x95, 0x0B, 0x42, 0xFA, 0xC3, 0x4E ,
	0x08, 0x2E, 0xA1, 0x66, 0x28, 0xD9, 0x24, 0xB2, 0x76, 0x5B, 0xA2, 0x49, 0x6D, 0x8B, 0xD1, 0x25 ,
	0x72, 0xF8, 0xF6, 0x64, 0x86, 0x68, 0x98, 0x16, 0xD4, 0xA4, 0x5C, 0xCC, 0x5D, 0x65, 0xB6, 0x92 ,
	0x6C, 0x70, 0x48, 0x50, 0xFD, 0xED, 0xB9, 0xDA, 0x5E, 0x15, 0x46, 0x57, 0xA7, 0x8D, 0x9D, 0x84 ,
	0x90, 0xD8, 0xAB, 0x00, 0x8C, 0xBC, 0xD3, 0x0A, 0xF7, 0xE4, 0x58, 0x05, 0xB8, 0xB3, 0x45, 0x06 ,
	0xD0, 0x2C, 0x1E, 0x8F, 0xCA, 0x3F, 0x0F, 0x02, 0xC1, 0xAF, 0xBD, 0x03, 0x01, 0x13, 0x8A, 0x6B ,
	0x3A, 0x91, 0x11, 0x41, 0x4F, 0x67, 0xDC, 0xEA, 0x97, 0xF2, 0xCF, 0xCE, 0xF0, 0xB4, 0xE6, 0x73 ,
	0x96, 0xAC, 0x74, 0x22, 0xE7, 0xAD, 0x35, 0x85, 0xE2, 0xF9, 0x37, 0xE8, 0x1C, 0x75, 0xDF, 0x6E ,
	0x47, 0xF1, 0x1A, 0x71, 0x1D, 0x29, 0xC5, 0x89, 0x6F, 0xB7, 0x62, 0x0E, 0xAA, 0x18, 0xBE, 0x1B ,
	0xFC, 0x56, 0x3E, 0x4B, 0xC6, 0xD2, 0x79, 0x20, 0x9A, 0xDB, 0xC0, 0xFE, 0x78, 0xCD, 0x5A, 0xF4 ,
	0x1F, 0xDD, 0xA8, 0x33, 0x88, 0x07, 0xC7, 0x31, 0xB1, 0x12, 0x10, 0x59, 0x27, 0x80, 0xEC, 0x5F ,
	0x60, 0x51, 0x7F, 0xA9, 0x19, 0xB5, 0x4A, 0x0D, 0x2D, 0xE5, 0x7A, 0x9F, 0x93, 0xC9, 0x9C, 0xEF ,
	0xA0, 0xE0, 0x3B, 0x4D, 0xAE, 0x2A, 0xF5, 0xB0, 0xC8, 0xEB, 0xBB, 0x3C, 0x83, 0x53, 0x99, 0x61 ,
	0x17, 0x2B, 0x04, 0x7E, 0xBA, 0x77, 0xD6, 0x26, 0xE1, 0x69, 0x14, 0x63, 0x55, 0x21, 0x0C, 0x7D
};

__constant__ unsigned char MixCol_dev[4][4] =
{
	{ 0x02,0x03,0x01,0x01 },
	{ 0x01,0x02,0x03,0x01 },
	{ 0x01,0x01,0x02,0x03 },
	{ 0x03,0x01,0x01,0x02 }
};

__constant__ unsigned char InvMixCol_dev[4][4] = {
	{ 0x0e, 0x0b, 0x0d, 0x09 },
	{ 0x09, 0x0e, 0x0b, 0x0d },
	{ 0x0d, 0x09, 0x0e, 0x0b },
	{ 0x0b, 0x0d, 0x09, 0x0e }
};


//----------------------------------------
typedef struct {
	unsigned char item[4][4];
} Block;


//----------------------------------------
// file length in number of characters
__host__ long file_length(const char* filename) {
	FILE * f = fopen(filename, "r");
	long length;
	if (f)
	{
		fseek(f, 0, SEEK_END);
		length = ftell(f);
		fclose(f);
		return length;
	}
	else
		return 0;
}
//----------------------------------------
// KEY SCHEDULING ALGORITHM
__host__ void key_scheduling(Block * keys) {
	
	//initial key
	unsigned char key[4][4] = {
		{ 0x54, 0x73, 0x20, 0x67 },
		{ 0x68, 0x20, 0x4b, 0x20 },
		{ 0x61, 0x6d, 0x75, 0x46 },
		{ 0x74, 0x79, 0x6e, 0x75 } };

	for (int i = 0; i < 4; i++)
	{
		for (int j = 0; j < 4; j++)
		{
			keys[0].item[i][j] = key[i][j];
		}
	}

	// key scheduling algorithm
	for (int k = 1; k <= 10; k++) {
		Block tempNew;
		Block tempOld = keys[k - 1];
		unsigned int temp[4] = { 
			tempOld.item[0][3], 
			tempOld.item[1][3], 
			tempOld.item[2][3], 
			tempOld.item[3][3] 
		};
	// ROTWORD
		unsigned int t;
		t = temp[0];
		temp[0] = temp[1];
		temp[1] = temp[2];
		temp[2] = temp[3];
		temp[3] = t;

		//SUBBYTES
		for (int i = 0; i < 4; i++) {
			temp[i] = Sbox1[temp[i]];
		}
		unsigned int temp2[4] = { 
			tempOld.item[0][0], 
			tempOld.item[1][0], 
			tempOld.item[2][0], 
			tempOld.item[3][0] };

		//RCON	
		//xor second column and temp and Rcon 1st round
		for (int i = 0; i < 4; i++) {
			temp2[i] = temp[i] ^ tempOld.item[i][0];
			temp2[i] = temp2[i] ^ Rcon[i][k - 1];
		}
		for (int i = 0; i < 4; i++)  //first column of 2nd key
			tempNew.item[i][0] = temp2[i];
		for (int j = 1; j < 4; j++) {
			for (int i = 0; i < 4; i++)
			{
				tempNew.item[i][j] = (tempNew.item[i][j - 1] ^ tempOld.item[i][j]);
			}
		}
		keys[k] = tempNew;
	} //end of key scheduling
}
//----------------------------------------

__host__ void real_initialization(Block * in, Block * out, int real_num_of_blocks) {
	for (size_t t = 0; t < real_num_of_blocks; t++)
	{
		for (size_t i = 0; i < 4; i++)
		{
			for (size_t j = 0; j < 4; j++)
			{
				out[t].item[i][j] = in[t].item[i][j];
			}
		}
	}
}

//----------------------------------------
// reading text file and initialization of array of blocks
__host__ int  initialization(char* source, Block * plaintext, int num_of_blocks) {
	Block * temp;
	int num = 0;
	cudaMallocHost((void**)&temp, num_of_blocks * sizeof(Block));
	ifstream ifs(source);
	int k = 0;
	while (ifs) {
		ifs.read((char *)temp[k].item, 16);
		k++;
	}
	for (size_t t = 0; t < num_of_blocks; t++)
	{
		for (size_t i = 0; i < 4; i++)
		{
			for (size_t j = 0; j < 4; j++)
			{
				if (temp[t].item[i][j] == '\n')
					num++;
				plaintext[t].item[j][i] = temp[t].item[i][j];
			}
		}
	}
	cudaFree(temp);
	return num;
}
//----------------------------------------
// printing blocks to stdout
__host__ void printBlocks(Block * in, int num_of_blocks) {
	for (size_t t = 0; t < num_of_blocks; t++)
	{
		fprintf(stdout, "%d. block\n", t);
		for (size_t i = 0; i < 4; i++)
		{
			for (size_t j = 0; j < 4; j++)
			{
				fprintf(stdout, " %0x", in[t].item[i][j]);
			}
			fprintf(stdout, "\n");
		}
		fprintf(stdout, "-----------------\n");
	}
}
//----------------------------------------
// writing array of blocks into text file
__host__ void writeToFile(Block * in, int num_of_blocks, char * filename) {
	FILE * file = fopen(filename, "w");
	if (file) {
		for (size_t t = 0; t < num_of_blocks; t++)
		{
			for (size_t i = 0; i < 4; i++)
			{
				for (size_t j = 0; j < 4; j++)
				{
					if (in[t].item[j][i] != 0)
						fprintf(file, "%c", in[t].item[j][i]);
				}
			}
		}
	}
	else
		fprintf(stdout, "Error opening file %s ", filename);
}

// new mix columns function
__device__ unsigned int mc(unsigned int a, unsigned int b, unsigned int c, unsigned int d)
{
	unsigned int Tmp, Tm, e;
	Tmp = a ^ b ^ c ^ d;
	Tm = a ^ b;
	Tm = xtime(Tm);
	e = Tm ^ Tmp ^ a;
	return e;
}
//----------------------------------------
// mix columns 
__device__ unsigned int mixColumns(unsigned int m0,
	unsigned int m1,
	unsigned int m2,
	unsigned int m3,
	unsigned int c0,
	unsigned int c1,
	unsigned int c2,
	unsigned int c3) {
	unsigned int rez0 = 0;
	unsigned int rez1 = 0;
	unsigned int rez2 = 0;
	unsigned int rez3 = 0;

	switch (m0)
	{
	case 1:
		rez0 = c0;
		break;
	case 2:
		rez0 = c0 << 1;
		if ((((c0 & 0x80) >> 7) & 0x01) == 1)
			rez0 ^= 0x1b;
		break;
	case 3:
		rez0 = c0 << 1;
		if ((((c0 & 0x80) >> 7) & 0x01) == 1)
			rez0 ^= 0x1b;
		rez0 ^= c0;
		break;
	default:
		break;
	}
	switch (m1)
	{
	case 1:
		rez1 = c1;
		break;
	case 2:
		rez1 = c1 << 1;
		if ((((c1 & 0x80) >> 7) & 0x01) == 1)
			rez1 ^= 0x1b;
		break;
	case 3:
		rez1 = c1 << 1;
		if ((((c1 & 0x80) >> 7) & 0x01) == 1)
			rez1 ^= 0x1b;
		rez1 ^= c1;
		break;
	default:
		break;
	}
	switch (m2)
	{
	case 1:
		rez2 = c2;
		break;
	case 2:
		rez2 = c2 << 1;
		if ((((c2 & 0x80) >> 7) & 0x01) == 1)
			rez2 ^= 0x1b;
		break;
	case 3:
		rez2 = c2 << 1;
		if ((((c2 & 0x80) >> 7) & 0x01) == 1)
			rez2 ^= 0x1b;
		rez2 ^= c2;
		break;
	default:
		break;
	}
	switch (m3)
	{
	case 1:
		rez3 = c3;
		break;
	case 2:
		rez3 = c3 << 1;
		if ((((c3 & 0x80) >> 7) & 0x01) == 1)
			rez3 ^= 0x1b;
		break;
	case 3:
		rez3 = c3 << 1;
		if ((((c3 & 0x80) >> 7) & 0x01) == 1)
			rez3 ^= 0x1b;
		rez3 ^= c3;
		break;
	default:
		break;
	}
	return rez0 ^ rez1 ^ rez2 ^ rez3;
}
//----------------------------------------
// inverse mix columns
__device__ unsigned int inverseMixColumns(unsigned int in0,
	unsigned int in1,
	unsigned int in2,
	unsigned int in3,
	unsigned int p0,
	unsigned int p1,
	unsigned int p2,
	unsigned int p3) {
	return Multiply(p0, in0) ^ Multiply(p1, in1) ^ Multiply(p2, in2) ^ Multiply(p3, in3);
}
//----------------------------------------
// encryption kernel
__global__ void encrypt(Block *keys, Block *plaintext, Block *ciphertext, unsigned int num_of_blocks)
{
	int i = threadIdx.x;
	int j = threadIdx.y;
	int blockNumber = blockIdx.y;

	//initial add round key
	ciphertext[blockNumber].item[i][j] = keys[0].item[i][j] ^ plaintext[blockNumber].item[i][j];
	//__syncthreads();

	for (int k = 1; k < 10; k++)
	{
		//subBytes
		ciphertext[blockNumber].item[i][j] = Sbox_dev[ciphertext[blockNumber].item[i][j]];
		//__syncthreads();

		//shift rows
		ciphertext[blockNumber].item[i][j] = ciphertext[blockNumber].item[i][(j + i) % 4];
		//__syncthreads();

		//mixColumns
		ciphertext[blockNumber].item[i][j] = mc(ciphertext[blockNumber].item[i][j],
			ciphertext[blockNumber].item[(i+1)%4][j],
			ciphertext[blockNumber].item[(i+2)%4][j],
			ciphertext[blockNumber].item[(i+3)%4][j]);

		//add round key
		ciphertext[blockNumber].item[i][j] = keys[k].item[i][j] ^ ciphertext[blockNumber].item[i][j];
		//__syncthreads();
	}
	//subbytes
	ciphertext[blockNumber].item[i][j] = Sbox_dev[ciphertext[blockNumber].item[i][j]];
	//__syncthreads();

	//rotwords
	ciphertext[blockNumber].item[i][j] = ciphertext[blockNumber].item[i][(j + i) % 4];
	//__syncthreads();

	//add round key
	ciphertext[blockNumber].item[i][j] = keys[10].item[i][j] ^ ciphertext[blockNumber].item[i][j];
	//__syncthreads();
}
//----------------------------------------
// decryption kernel
__global__ void decrypt(Block *keys, Block *plaintext, Block *ciphertext, unsigned int num_of_blocks) {

	int i = threadIdx.x;
	int j = threadIdx.y;
	int blockNumber = blockIdx.y;

	//inverse add round key
	plaintext[blockNumber].item[i][j] = ciphertext[blockNumber].item[i][j] ^ keys[10].item[i][j];
	//__syncthreads();

	for (size_t k = 9; k >= 1; k--)
	{
		//inverse shift rows
		plaintext[blockNumber].item[i][j] = plaintext[blockNumber].item[i][(4 + j - i) % 4];
		//__syncthreads();

		//inverse subbytes
		plaintext[blockNumber].item[i][j] = InvSbox_dev[plaintext[blockNumber].item[i][j]];
		//__syncthreads();

		//inverse add round key
		plaintext[blockNumber].item[i][j] = plaintext[blockNumber].item[i][j] ^ keys[k].item[i][j];
		//__syncthreads();


		//inverse mixColumns
		plaintext[blockNumber].item[i][j] = inverseMixColumns(
			InvMixCol_dev[i][0],
			InvMixCol_dev[i][1],
			InvMixCol_dev[i][2],
			InvMixCol_dev[i][3],
			plaintext[blockNumber].item[0][j],
			plaintext[blockNumber].item[1][j],
			plaintext[blockNumber].item[2][j],
			plaintext[blockNumber].item[3][j]);
		//__syncthreads();


	}
	//inverse shift rows
	plaintext[blockNumber].item[i][j] = plaintext[blockNumber].item[i][(4 + j - i) % 4];
	//__syncthreads();

	//inverse subbytes
	plaintext[blockNumber].item[i][j] = InvSbox_dev[plaintext[blockNumber].item[i][j]];
	//__syncthreads();

	//inverse add round key
	plaintext[blockNumber].item[i][j] = plaintext[blockNumber].item[i][j] ^ keys[0].item[i][j];
	//__syncthreads();
}
//----------------------------------------
// main function
int main()
{
	cudaDeviceProp deviceProp;
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
	}
	//----------------------- device properties ------------------------------------
	cudaStatus = cudaGetDeviceProperties(&deviceProp, 0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cudaGetProp failed!");
	}
	fprintf(stdout, "   Hardware information:\n");
	fprintf(stdout, "     Device name:                  %s\n", deviceProp.name);
	fprintf(stdout, "     Major revision number:        %d\n", deviceProp.major);
	fprintf(stdout, "     Minor revision Number:        %d\n", deviceProp.minor);
	fprintf(stdout, "     Memory clock rate:            %d  MHz\n", deviceProp.memoryClockRate / 1000);
	fprintf(stdout, "     Clock Rate:                   %d MHz\n", deviceProp.clockRate / 1000);
	fprintf(stdout, "     Total Global Memory:          %d MB\n", deviceProp.totalGlobalMem / 1024 / 1024);
	fprintf(stdout, "     L2 Cache memory size:         %d  KB\n", deviceProp.l2CacheSize / 1024);
	fprintf(stdout, "     Total shared mem per block:   %d   KB\n", deviceProp.sharedMemPerBlock / 1024);
	fprintf(stdout, "     Total const mem size:         %d   KB\n", deviceProp.totalConstMem / 1024);
	fprintf(stdout, "     Warp size:                    %d\n", deviceProp.warpSize);
	fprintf(stdout, "     Maximum block dimensions:     %d x %d x %d\n", deviceProp.maxThreadsDim[0],
		deviceProp.maxThreadsDim[1],
		deviceProp.maxThreadsDim[2]);
	fprintf(stdout, "     Maximum grid dimensions:      %d x %d x %d\n", deviceProp.maxGridSize[0],
		deviceProp.maxGridSize[1],
		deviceProp.maxGridSize[2]);
	fprintf(stdout, "     Number of muliprocessors:     %d\n", deviceProp.multiProcessorCount);
	fprintf(stdout, "     Max threads per block:        %d\n", deviceProp.maxThreadsPerBlock);
	fprintf(stdout, "     Supports conncurent kernels:  %s\n\n\n", (deviceProp.concurrentKernels == 1) ? "Yes" : "No");

	ofstream off3;
	Block * keys;
	cudaStatus = cudaMallocHost((void**)&keys, 11 * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Keys allocation error #%d\n", cudaStatus);
	}
	key_scheduling(keys);

	fprintf(stdout, "   Application information\n");
	cout << "     Enter text file name: ";
	char name[20];
	scanf("%s", name);

	long plaintext_length = file_length(name);
	long num_of_blocks = (plaintext_length % 16 == 0) ? plaintext_length / 16 : plaintext_length / 16 + 1;

	fprintf(stdout, "     Plaintext length: %ld characters \n", plaintext_length);

	Block * plaintext;
	cudaStatus = cudaMallocHost((void**)&plaintext, num_of_blocks * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Plaintext allocation error #%d\n", cudaStatus);
	}
	int num_of_zeros = initialization(name, plaintext, num_of_blocks);
	int real_num_of_block = ((plaintext_length - num_of_zeros) % 16 == 0) ? (plaintext_length - num_of_zeros) / 16 : (plaintext_length - num_of_zeros) / 16 + 1;
	fprintf(stdout, "     Number of blocks: %d \n", num_of_blocks);
	fprintf(stdout, "     Number of real blocks: %d \n", real_num_of_block);

	Block * real_plaintext;
	cudaStatus = cudaMallocHost((void**)&real_plaintext, real_num_of_block * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Real plaintext allocation error #%d\n", cudaStatus);
	}
	cudaMemcpy(real_plaintext, plaintext, real_num_of_block * sizeof(Block), cudaMemcpyHostToHost);
	cudaFree(plaintext);

	Block * ciphertext;
	cudaStatus = cudaMallocHost((void**)&ciphertext, real_num_of_block * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Ciphertext allocation error #%d\n", cudaStatus);
	}
	Block * plaintext2;
	cudaStatus = cudaMallocHost((void**)&plaintext2, real_num_of_block * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Plaintext2 allocation error #%d\n", cudaStatus);
	}

	//--------------- device memory allocation -------------------
	cudaEvent_t startAlloc, stopAlloc;
	float timeAlloc;
	cudaEventCreate(&startAlloc);
	cudaEventCreate(&stopAlloc);
	cudaEventRecord(startAlloc, 0);

	Block *keys_dev;
	cudaStatus = cudaMalloc((void**)&keys_dev, 11 * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Allocating memory for key blocks failed!");
		goto Error;
	}
	cudaStatus = cudaMemset(keys_dev, 0, 11 * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "keys memset error #%d\n", cudaStatus);
	}

	Block *plaintext_dev;
	cudaStatus = cudaMalloc((void**)&plaintext_dev, real_num_of_block * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Allocating memory for plaintext blocks failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(plaintext_dev, 0, real_num_of_block * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "plaintext memset error #%d\n", cudaStatus);
	}

	Block *ciphertext_dev;
	cudaStatus = cudaMalloc((void**)&ciphertext_dev, real_num_of_block * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Allocating memory for ciphertext blocks failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(ciphertext_dev, 0, real_num_of_block * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "cipher memset error #%d\n", cudaStatus);
	}

	Block* plaintext2_dev;
	cudaStatus = cudaMalloc((void**)&plaintext2_dev, real_num_of_block * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Allocating memory for decrypted text blocks failed!");
		goto Error;
	}

	cudaStatus = cudaMemset(plaintext2_dev, 0, real_num_of_block * sizeof(Block));
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "plaintext2 memset error #%d\n", cudaStatus);
	}
	//------------------ copying block from host to device -----------------
	cudaStatus = cudaMemcpy(keys_dev, keys, 11 * sizeof(Block), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Copying key blocks on device failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(plaintext_dev, real_plaintext, real_num_of_block * sizeof(Block), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Copying plaintext blocks on device failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(ciphertext_dev, ciphertext, real_num_of_block * sizeof(Block), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Copying ciphertext blocks on device failed!");
		goto Error;
	}

	cudaStatus = cudaMemcpy(plaintext2_dev, plaintext2, real_num_of_block * sizeof(Block), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Copying decrypted blocks on device failed!");
		goto Error;
	}
	cudaEventRecord(stopAlloc, 0);
	cudaEventSynchronize(stopAlloc);
	cudaEventElapsedTime(&timeAlloc, startAlloc, stopAlloc);
	
	//----------------------------------------------------------------------
	cudaFree(keys);
	cudaFree(real_plaintext);

	int kernel_rounds = real_num_of_block / 50000;
	int rest = real_num_of_block - (kernel_rounds * 50000);

	dim3 threadsPerBlock(4, 4);
	dim3 numBlocks(1, 50000);
	dim3 numBlocks2(1, rest);
	fprintf(stdout, "\n   Encryption.........\n");
	cudaEvent_t startEnc, stopEnc;
	float timeEnc;
	cudaEventCreate(&startEnc);
	cudaEventCreate(&stopEnc);
//	int num_of_rounds;
	cudaEventRecord(startEnc, 0);
	for (size_t i = 0; i < kernel_rounds; i++)
	{
		encrypt << <numBlocks, threadsPerBlock >> > (keys_dev, plaintext_dev+i*50000, ciphertext_dev+i* 50000, 50000);
	}
	encrypt << <numBlocks2, threadsPerBlock >> > (keys_dev, plaintext_dev + kernel_rounds * 50000, ciphertext_dev + kernel_rounds * 50000, rest);
	cudaEventRecord(stopEnc, 0);
	cudaEventSynchronize(stopEnc);
	cudaEventElapsedTime(&timeEnc, startEnc, stopEnc);
	fprintf(stdout, "     Encryption time %.2f ms\n\n", timeEnc);

	//------------------ decryption --------------------------------------
	fprintf(stdout, "   Decryption.......\n");
	cudaEvent_t startDec, stopDec;
	float timeDec;
	cudaEventCreate(&startDec);
	cudaEventCreate(&stopDec);
	cudaEventRecord(startDec, 0);

	for (size_t i = 0; i < kernel_rounds; i++)
	{
		decrypt << <numBlocks, threadsPerBlock >> > (keys_dev, plaintext2_dev + i * 50000, ciphertext_dev + i * 50000, 50000);
	}
	decrypt << <numBlocks2, threadsPerBlock >> > (keys_dev, plaintext2_dev + kernel_rounds * 50000, ciphertext_dev + kernel_rounds * 50000, rest);
	cudaEventRecord(stopDec, 0);
	cudaEventSynchronize(stopDec);
	cudaEventElapsedTime(&timeDec, startDec, stopDec);
	fprintf(stdout, "     Decryption time %.2f ms\n", timeDec);
	cudaStatus = cudaMemcpy(ciphertext, ciphertext_dev, real_num_of_block * sizeof(Block), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Copying ciphertext blocks on host failed!");
		goto Error;
	}
	cudaEvent_t startDealloc, stopDealloc;
	float timeDealloc;
	cudaEventCreate(&startDealloc);
	cudaEventCreate(&stopDealloc);
	cudaEventRecord(startDealloc, 0);
	cudaStatus = cudaMemcpy(plaintext2, plaintext2_dev, real_num_of_block * sizeof(Block), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stdout, "Copying decrypted blocks on host failed!");
		goto Error;
	}
	cudaEventRecord(stopDealloc, 0);
	cudaEventSynchronize(stopDealloc);
	cudaEventElapsedTime(&timeDealloc, startDealloc, stopDealloc);

	int index = 0;
	for (size_t i = 0; i < 20; i++)
	{
		if (name[i] == '.')
		{
			index = i;
			break;
		}
	}
	
	name[index] = '_';
	name[index + 1] = 'i';
	name[index + 2] = 'n';
	name[index + 3] = 'f';
	name[index + 4] = 'o';
	name[index + 5] = '.';
	name[index + 6] = 't';
	name[index + 7] = 'x';
	name[index + 8] = 't';
	name[index + 9] = '\0';
	off3.open(name, ofstream::trunc);
	off3 << "Number of characters in file: " << plaintext_length << endl;
	off3 << "Number of blocks with zeros: " << num_of_blocks << endl;
	off3 << "Number of zeros: " << num_of_zeros << endl;
	off3 << "Number of blocks without zeros: " << real_num_of_block << endl;
	off3 << "Encryption finished in " << timeEnc << " miliseconds" << endl;
	off3 << "Decryption finished in " << timeDec << " miliseconds" << endl;
	off3.close();

	fprintf(stdout, "\n   Writing to files....\n\n");
	//writeToFile(ciphertext, real_num_of_block, "encrypted_text.txt");
	//writeToFile(plaintext2, real_num_of_block, "decrypted_text.txt");
Error:
	cudaEvent_t startDealloc2, stopDealloc2;
	float timeDealloc2;
	cudaEventCreate(&startDealloc2);
	cudaEventCreate(&stopDealloc2);
	cudaEventRecord(startDealloc2, 0);

	cudaFree(ciphertext);
	cudaFree(plaintext2);
	cudaFree(keys_dev);
	cudaFree(plaintext_dev);
	cudaFree(ciphertext_dev);

	cudaEventRecord(stopDealloc2, 0);
	cudaEventSynchronize(stopDealloc2);
	cudaEventElapsedTime(&timeDealloc2, startDealloc2, stopDealloc2);

	//fprintf(stdout, "\nPress enter to end......");
	fprintf(stdout, "\n\nAllocation and copying time %.2f ms\n", timeAlloc);
	fprintf(stdout, "Deallocation and copying time %.2f ms\n\n", timeDealloc + timeDealloc2);
	return 0;
}