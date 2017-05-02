#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cufft.h>
#include <sndfile.h>

#define	BLOCK_SIZE 4096
//#define	BLOCK_SIZE 256 

static void print_usage (char *progname)
{
	printf("\nUsage : %s [--full-precision] <input file> <output file>\n", progname);
	puts ("\nWhere the output file will contain a line for each frame\n   and a column for each channel.\n");
}


/*
https://devtalk.nvidia.com/default/topic/401619/cuda-noob-quot-cufft-error-cufft_invalid_plan_FFT-quot-/?offset=5
https://devtalk.nvidia.com/default/topic/371911/cufft-error-cufft_invalid_plan_FFT/

===== Important ============   https://devtalk.nvidia.com/default/topic/410009/cufftexecr2c-only-gives-half-the-answer-33-/


The code should do something like:

1) allocate GPU memory

2) copy from CPU to GPU

3) call cuFFT ( input and output should be arrays in GPU memory)

4) copy result from GPU to CPU

5) free GPU memory

*/

__global__ void multiply(cufftComplex *songFD_GPU, cufftComplex *impulseFD_GPU)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < BLOCK_SIZE)
	{
		//songFD_GPU[i].x = (songFD_GPU[i].x * impulseFD_GPU[i].x - songFD_GPU[i].y * impulseFD_GPU[i].y) * 1./BLOCK_SIZE;
		//songFD_GPU[i].x = (songFD_GPU[i].x * impulseFD_GPU[i].y + songFD_GPU[i].y * impulseFD_GPU[i].x) * 1./BLOCK_SIZE;
		songFD_GPU[i].x = (songFD_GPU[i].x * impulseFD_GPU[i].x - songFD_GPU[i].y * impulseFD_GPU[i].y);
		songFD_GPU[i].x = (songFD_GPU[i].x * impulseFD_GPU[i].y + songFD_GPU[i].y * impulseFD_GPU[i].x);
	}
}


static void convert_to_text_CUDA (SNDFILE * infile, SNDFILE * impulsefile, SNDFILE * finalfile, int channels, int full_precision)
{	
	cudaError_t err = cudaSuccess;
	cufftResult result;	
	cufftHandle plan_FFT;
	cufftHandle plan_IFFT;
	int readcount = 0;

	// Allocate memory for Realdata at Host side	
	cufftReal buf[BLOCK_SIZE], bufI[BLOCK_SIZE], in2[BLOCK_SIZE];
	// Allocate memory for Realdata at GPU side
	cufftReal *buf_GPU, *bufI_GPU, *in2_GPU;
	err = cudaMalloc((void **)&buf_GPU, sizeof(cufftReal)*(BLOCK_SIZE)); 
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		return;
	}
	err = cudaMalloc((void **)&bufI_GPU, sizeof(cufftReal)*(BLOCK_SIZE)); 
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		return;
	}
	
	err = cudaMalloc((void **)&in2_GPU, sizeof(cufftReal)*(BLOCK_SIZE)); 
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
		return;
	}
	

	// Allocate memory for Complexdata at Host side
	cufftComplex songFD[BLOCK_SIZE], impulseFD[BLOCK_SIZE];
	// Allocate memory for Complexdata at GPU side
	cufftComplex *songFD_GPU, *impulseFD_GPU;
	err = cudaMalloc((void**)&songFD_GPU, sizeof(cufftComplex)*(BLOCK_SIZE));
	if (err != cudaSuccess)
	{
		printf("Cuda error: Failed to allocate for song\n");
		return;	
	}
	
	err = cudaMalloc((void**)&impulseFD_GPU, sizeof(cufftComplex)*(BLOCK_SIZE));
	if (err != cudaSuccess)
	{
		printf("Cuda error: Failed to allocate for impulse\n");
		return;	
	}


	// Create plan_FFT
	result = cufftPlan1d(&plan_FFT, BLOCK_SIZE, CUFFT_R2C, 1);
	if (result != CUFFT_SUCCESS)
	{
		printf("CUFFT error: Plan FFT creation failed\n");
		return;
	}
	
	// Create plan_IFFT
	result = cufftPlan1d(&plan_IFFT, BLOCK_SIZE, CUFFT_C2R, 1);
	if (result != CUFFT_SUCCESS)
	{
		printf("CUFFT error: Plan IFFT creation failed\n");
		return;
	}
	
	
	// Read song from audio file
	while ((readcount = sf_readf_float (infile, buf, BLOCK_SIZE)) > 0)
	{
		// Read impulse from audio file
		sf_readf_float (impulsefile, bufI, BLOCK_SIZE);

		// Copy dat from buf to buf_GPU
		err = cudaMemcpy(buf_GPU, buf, sizeof(cufftReal)*(BLOCK_SIZE), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			return;
		} 
		// Copy dat from bufI to bufI_GPU
		err = cudaMemcpy(bufI_GPU, bufI, sizeof(cufftReal)*(BLOCK_SIZE), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			return;
		}
	
        	//Song T -> F
		result = cufftExecR2C(plan_FFT, buf_GPU, songFD_GPU);
		if (result != CUFFT_SUCCESS)
		{
			printf("CUFFT error: ExecR2C failed for song\n");
			return;
		}
		
		//Impulse T -> F
		result = cufftExecR2C(plan_FFT, bufI_GPU, impulseFD_GPU);
		if (result != CUFFT_SUCCESS)
		{
			printf("CUFFT error: ExecR2C failed for impulse\n");
			return;
		}

		int blocksPerGrid = 4;
		int threadsPerBlock = 1024;
		multiply<<<blocksPerGrid,threadsPerBlock>>>(songFD_GPU, impulseFD_GPU);			
	
		//Song F -> T
		result = cufftExecC2R(plan_IFFT, songFD_GPU, in2_GPU);
		if (result != CUFFT_SUCCESS)
		{	
			printf("CUFFT error: ExecC2R failed for FINAL!!!\n");
			return;
		}	
		
		// Copy convolved song from GPU to HOST
		err = cudaMemcpy(in2, in2_GPU, sizeof(cufftReal)*(BLOCK_SIZE), cudaMemcpyDeviceToHost);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
			return;
		} 
	
		// *********** NORMALIZATION FU***** IS VERY VERY IMPORTANT !!!!!!*******************	
		for(int i = 0; i < BLOCK_SIZE; i++)
		{
			in2[i] /= (BLOCK_SIZE);
		}	

		//Store convolved song in new file
		sf_write_float(finalfile, in2, BLOCK_SIZE);
	}
}

int
main (int argc, char * argv [])
{	
	char 		*progname, *infilename, *impulsefilename, *finalfilename;
	SNDFILE		*infile = NULL ;
	SNDFILE		*impulsefile = NULL ;
	SNDFILE		*finalfile = NULL ;
	SF_INFO		sfinfo ;
	SF_INFO		sfinfoI ;
	SF_INFO		sfinfoO ;
	int		full_precision = 0 ;

	progname = strrchr (argv [0], '/') ;
	progname = progname ? progname + 1 : argv [0] ;

	switch (argc)
	{
		case 5 :
		if (!strcmp ("--full-precision", argv [5]))
		{
			print_usage (progname) ;
			return 1 ;
		} ;
		full_precision = 1 ;
		argv++ ;
		
		case 4 :
			break ;
		default:
			print_usage (progname) ;
			return 1 ;
	} ;

	infilename = argv [1] ;
    	impulsefilename = argv[2];
    	finalfilename = argv[3];

	if (infilename[0] == '-')
	{
		printf("Error : Input filename (%s) looks like an option.\n\n", infilename);
		print_usage(progname);
		return 1;
	}

    	if (impulsefilename[0] == '-')
	{	
		printf("Error : Input impulse filename (%s) looks like an option.\n\n", impulsefilename);
		print_usage (progname);
		return 1;
	}
	
    	if (finalfilename[0] == '-')
	{	
		printf ("Error : Final output filename (%s) looks like an option.\n\n", finalfilename) ;
		print_usage (progname) ;
		return 1;
	}

	memset (&sfinfo, 0, sizeof (sfinfo));
	memset (&sfinfoI, 0, sizeof (sfinfoI));
	memset (&sfinfoO, 0, sizeof (sfinfoO));

	if ((infile = sf_open (infilename, SFM_READ, &sfinfo)) == NULL)
	{
		printf("Not able to open input file %s.\n", infilename);
		puts(sf_strerror (NULL));
		return 1;
	}
	
	if ((impulsefile = sf_open (impulsefilename, SFM_READ, &sfinfoI)) == NULL)
	{
		printf("Not able to open input file %s.\n", impulsefilename);
		puts(sf_strerror(NULL));
		return 1 ;
	}

        sfinfoO.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        sfinfoO.channels = sfinfo.channels;
        sfinfoO.samplerate = sfinfo.samplerate;


	if ((finalfile = sf_open(finalfilename, SFM_WRITE, &sfinfoO)) == NULL)
	{
		printf("Not able to open input file %s.\n", finalfilename);
		puts(sf_strerror (NULL));
		return 1 ;
	}

	convert_to_text_CUDA (infile, impulsefile, finalfile, sfinfo.channels, full_precision);

	sf_close (infile);
	sf_close (impulsefile);
	sf_close (finalfile);

	return 0;
}

