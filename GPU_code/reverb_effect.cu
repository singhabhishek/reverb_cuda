#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <sndfile.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/time.h> 

//#define WITH_STREAM 1
#define WITHOUTSTREAM 1
#define LOUDEST_SAMPLE 51.344872
#define NORMALIZE_RATIO 1.5
#define THREADSPERBLOCK 768
//#define CPU 1

void channel_split(float* buffer, int num_frames, float** chan_buffers, int num_channels)
{
    int i;
    int samples = num_frames * num_channels;
    for (i = 0; i < samples; i++)
    {
        chan_buffers[(i % num_channels)][i/num_channels] = buffer[i];
    }
}

__global__ void channel_split_kernel(float* in, float *ch1, float *ch2, int num_frames)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < num_frames)
	{
       	i%2?(ch2[i/2 + 1]=in[i]):(ch1[i/2] = in[i]); 
    }
}

void channel_join(float** chan_buffers, int num_channels, float* buffer, int num_frames)
{
    int i;
    int samples = num_frames * num_channels;
    for (i = 0; i < samples; i++)
    {
        buffer[i] = chan_buffers[i % num_channels][i / num_channels];
    }
}

__global__ void channel_join_kernel(float *ch1, float *ch2, float* buffer, int num_frames)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	if(i < num_frames)
	{
		buffer[2*i] = ch1[i];
		buffer[2*i + 1] = ch2[i];
	}
}

void normalize(float* buffer, int num_samps, float maxval)
{
    float loudest_sample = 0.0;
    float multiplier = 0.0;
    int i;

    for (i = 0; i < num_samps; i++)
    {
        if (fabs(buffer[i]) > loudest_sample) loudest_sample = buffer[i];
    }
    multiplier = maxval / loudest_sample;

    for (i = 0; i < num_samps; i++)
    {
        buffer[i] *= multiplier;
    }
}

__global__ void normalize_kernel(float * buffer, int num_samps, float maxval)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
    buffer[i] *= maxval;
}

__global__ void conv(float *A, float *B, float *out, int mask_width, int width)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;
	float Pvalue = 0.0;
	int N_start_point  = (i - (mask_width/2));
	if (i < width)
	{
		for (int j = 0; j < mask_width; j++)
		{
			if (((N_start_point + j) >= 0) && ((N_start_point + j) < width))
			{
				Pvalue += A[N_start_point + j]*B[j];
			}
		}
		out[i] = Pvalue;
	}
	__syncthreads();
}

__global__ void conv_shared(const float *d_Signal, const float *d_ConvKernel, float *d_Result_GPU, const int K, const int N)
{

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float d_Tile[THREADSPERBLOCK];

    d_Tile[threadIdx.x] = d_Signal[i];
    __syncthreads();

    float temp = 0.f;

    int N_start_point = i - (K / 2);

    for (int j = 0; j < K; j++) if (N_start_point + j >= 0 && N_start_point + j < N)
    {
            if ((N_start_point + j >= blockIdx.x * blockDim.x) && (N_start_point + j < (blockIdx.x + 1) * blockDim.x))
                temp += d_Tile[threadIdx.x + j - (K / 2)] * d_ConvKernel[j]; 
            else
                temp += d_Signal[N_start_point + j] * d_ConvKernel[j];
    }
    d_Result_GPU[i] = temp;
}

__global__ void conv_Kernel(float *A, float *B, float *C, int P, int N)
{
	int idx = threadIdx.x+blockDim.x*blockIdx.x;
	int radius = (P-1)/2;
	if ((idx < (N-radius)) && (idx >= radius)){
		float my_sum = 0;
		for (int j = -radius; j <= radius; j++)
		{
			my_sum += A[idx+j]*B[j+radius];
		}
		C[idx] = my_sum;
	}
}

int main(int argc, char** argv)
{
    // Initialize the variables
    enum {ARG_NAME, ARG_INFILE1, ARG_INFILE2, ARG_OUTFILE, ARG_NARGS};
    int i, k;
    time_t current_time, end_time;
    double process_time;
    SNDFILE* infile1;
    SNDFILE* infile2;
    SNDFILE* outfile;
    SF_INFO props1;
    SF_INFO props2;
    SF_INFO props3;
    int outFrames;
    float *b1;
    float *b2;
    float *b3;
    float** buf1;
    float** buf2;
    float** buf3;
    cudaError_t err = cudaSuccess;
	float *d_b3 = NULL;
	float *d_result = NULL;
	float *d_ch2 = NULL;
	float *d_ch1 = NULL;
	float *d_song = NULL;
	float *d_imp = NULL;
	float *d_conv = NULL;

    current_time = time(NULL);

    // Check the inputs
    if (argc != ARG_NARGS)
    {
        fprintf(stderr, "Error: incorrect number of arguments.\nUsage: %s infile1 infile2 outfile\n", argv[ARG_NAME]);
        return -1;
    }

    // Initialize the files
    infile1 = sf_open(argv[ARG_INFILE1], SFM_READ, &props1);
    infile2 = sf_open(argv[ARG_INFILE2], SFM_READ, &props2);

    if (infile1 == NULL)
    {
        fprintf(stderr, "Error opening %s\n", argv[ARG_INFILE1]);
        return -1;
    }

    if (infile2 == NULL)
    {
        fprintf(stderr, "Error opening %s\n", argv[ARG_INFILE2]);
        return -1;
    }
	
	if((props1.channels > 2) || (props2.channels > 2))
	{
        fprintf(stderr, "Only support mono/stereo audio\n");
		return -1;
	}
	
    if (props1.samplerate != props2.samplerate)
    {
        fprintf(stderr, "Error, samplerates must match:\n%s:\t%i\n%s:\t%i\n",
                argv[ARG_INFILE1], props1.samplerate,
                argv[ARG_INFILE2], props2.samplerate
               );
        return -1;
    }

    if (props1.channels != props2.channels)
    {
        fprintf(stderr, "Error, channels must match:\n%s:\t%i\n%s:\t%i\n",
                argv[ARG_INFILE1], props1.channels,
                argv[ARG_INFILE2], props2.channels
               );
        return -1;
    }

    // Let's make some buffers and the output file
    outFrames = (props1.frames + props2.frames - 1);
    props3 = props1;
    props3.frames = outFrames;
    outfile = sf_open(argv[ARG_OUTFILE], SFM_WRITE, &props3);
	props3.frames = (props1.frames + props2.frames - 1);

    buf1 = (float**)malloc(sizeof(float*) * props1.channels);
    buf2 = (float**)malloc(sizeof(float*) * props2.channels);
    buf3 = (float**)malloc(sizeof(float*) * props1.channels);

    for (i = 0; i < props1.channels; i++)
    {
        buf1[i] = (float*)malloc(sizeof(float) * props1.frames);
        buf2[i] = (float*)malloc(sizeof(float) * props2.frames);
        buf3[i] = (float*)malloc(sizeof(float) * outFrames);
    }

    b1 = (float*)malloc(sizeof(float) * props1.frames * props1.channels);
    b2 = (float*)malloc(sizeof(float) * props2.frames * props2.channels);
    b3 = (float*)malloc(sizeof(float) * outFrames * props1.channels);

    // Read
    sf_readf_float(infile1, b1, props1.frames);
    sf_readf_float(infile2, b2, props2.frames);

    if (!b1)
    {
        fprintf(stderr, "Error reading %s\n", argv[ARG_INFILE1]);
        return -1;
    }
    if (!b2)
    {
        fprintf(stderr, "Error reading %s\n", argv[ARG_INFILE2]);
        return -1;
    }

	struct timeval t1, t2;
	double elapsedTime;
	gettimeofday(&t1, NULL);

	// Split channels
#ifdef CPU
	printf("channel_split CPU\n");
    channel_split(b1, props1.frames, buf1, props1.channels);
	channel_split(b2, props2.frames, buf2, props2.channels);
#else
	printf("channel_split GPU\n");
	if(props1.channels == 1)
	{
		memcpy(buf1[0], b1, props1.frames*sizeof(float));
		memcpy(buf2[0], b2, props2.frames*sizeof(float));
	}
	else if(props1.channels == 2)
	{
		// Song
		float *d_s_in = NULL;
    	cudaMalloc((void **)&d_s_in, props1.channels*props1.frames*sizeof(float));
		cudaMemcpy(d_s_in, b1, props1.channels*props1.frames*sizeof(float), cudaMemcpyHostToDevice);
		float *d_s_ch1 = NULL;
    	cudaMalloc((void **)&d_s_ch1, props1.frames*sizeof(float));
		float *d_s_ch2 = NULL;
    	cudaMalloc((void **)&d_s_ch2, props1.frames*sizeof(float));

		// Impulse 
		float *d_i_in = NULL;
    	cudaMalloc((void **)&d_i_in, props2.channels*props2.frames*sizeof(float));
		cudaMemcpy(d_i_in, b2, props2.channels*props2.frames*sizeof(float), cudaMemcpyHostToDevice);
		float *d_i_ch1 = NULL;
    	cudaMalloc((void **)&d_i_ch1, props2.frames*sizeof(float));
		float *d_i_ch2 = NULL;
    	cudaMalloc((void **)&d_i_ch2, props2.frames*sizeof(float));
		
		channel_split_kernel<<<(props1.channels*props1.frames)/THREADSPERBLOCK,THREADSPERBLOCK>>>(d_s_in, d_s_ch1, d_s_ch2, props1.channels*props1.frames);
		channel_split_kernel<<<(props2.channels*props2.frames)/THREADSPERBLOCK,THREADSPERBLOCK>>>(d_i_in, d_i_ch1, d_i_ch2, props2.channels*props2.frames);
		
		cudaMemcpy(buf1[0], d_s_ch1, props1.frames*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(buf1[1], d_s_ch2, props1.frames*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(buf2[0], d_i_ch1, props2.frames*sizeof(float), cudaMemcpyDeviceToHost);
		cudaMemcpy(buf2[1], d_i_ch2, props2.frames*sizeof(float), cudaMemcpyDeviceToHost);
	}
#endif

	gettimeofday(&t2, NULL);
	elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;
	elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;
    printf("Convolution complete in %f ms.\n", elapsedTime);

    for (k = 0; k < props1.channels; k++)
		memset(buf3[k], outFrames, 0);

#ifdef WITH_STREAM
	printf("WITH STREAM!!!!\n"); 
	cudaStream_t stream0, stream1;
	cudaStreamCreate(&stream0);
	cudaStreamCreate(&stream1);

    float *d_A0, *d_B0, *d_C0;
    float *d_A1, *d_B1, *d_C1;

    cudaMalloc((void**)&d_A0,props1.frames*sizeof(float));
    cudaMalloc((void**)&d_B0,props2.frames*sizeof(float));
    cudaMalloc((void**)&d_C0,props3.frames*sizeof(float));
    cudaMalloc((void**)&d_A1,props1.frames*sizeof(float));
    cudaMalloc((void**)&d_B1,props2.frames*sizeof(float));
    cudaMalloc((void**)&d_C1,props3.frames*sizeof(float));

	// Copy Song data from Host to Device
	cudaMemcpyAsync(d_A0, buf1[0], props1.frames*sizeof(float), cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(d_B0, buf2[0], props2.frames*sizeof(float), cudaMemcpyHostToDevice, stream0);
	cudaMemcpyAsync(d_A1, buf1[1], props1.frames*sizeof(float), cudaMemcpyHostToDevice, stream1);
	cudaMemcpyAsync(d_B1, buf2[1], props2.frames*sizeof(float), cudaMemcpyHostToDevice, stream1);


	// Call Kernel for each channel
	printf("SONG = %lld\n", props1.frames);
	printf("IMP = %lld\n", props2.frames);
	printf("CV = %lld\n", props3.frames);
	printf("CUDA kernel launch with %d blocks of %d threads\n", (unsigned int)(props3.frames/THREADSPERBLOCK), THREADSPERBLOCK);
	conv<<<props3.frames/THREADSPERBLOCK,THREADSPERBLOCK,0,stream0>>>(d_A0, d_B0, d_C0, props2.frames, props3.frames);
	conv<<<props3.frames/THREADSPERBLOCK,THREADSPERBLOCK,0,stream1>>>(d_A1, d_B1, d_C1, props2.frames, props3.frames);
	//cudaDeviceSynchronize();
	cudaStreamSynchronize(stream0);
	cudaStreamSynchronize(stream1);

	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to launch vectorMultiplyAdd kernel (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
		
	// Copy convolved data from Device to Host
	cudaMemcpyAsync(buf3[0], d_C0, props3.frames*sizeof(float), cudaMemcpyDeviceToHost, stream0);
	cudaMemcpyAsync(buf3[1], d_C1, props3.frames*sizeof(float), cudaMemcpyDeviceToHost, stream1);
#endif

#ifdef WITHOUTSTREAM
	printf("WITHOUT STREAM!!!!\n"); 
	// Allocate memory for Song on device
    err = cudaMalloc((void **)&d_song, props1.frames*sizeof(float));
    if (err != cudaSuccess)
    {
		fprintf(stderr, "Failed to allocate device imp vector (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
    }
	
	// Allocate memory for Impulse on device
	err = cudaMalloc((void **)&d_imp, props2.frames*sizeof(float));
	if (err != cudaSuccess)
	{
		fprintf(stderr, "Failed to allocate device imp vector (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
	
	// Allocate memory for convolved data on device
    err = cudaMalloc((void **)&d_conv, props3.frames*sizeof(float));
    if (err != cudaSuccess)
    {
		fprintf(stderr, "Failed to allocate device imp vector (error code %s)!\n", cudaGetErrorString(err));
		exit(EXIT_FAILURE);
    }
   
    for (k = 0; k < props1.channels; k++)
	{
		// Copy Song data from Host to Device	
		err = cudaMemcpy(d_song, buf1[k], props2.frames*sizeof(float), cudaMemcpyHostToDevice);
    	if (err != cudaSuccess)
    	{
			fprintf(stderr, "Failed to copy imp vector from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
    	}
	
		// Copy Impulse data from Host to Device	
		err = cudaMemcpy(d_imp, buf2[k], props2.frames*sizeof(float), cudaMemcpyHostToDevice);
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to copy imp vector from host to device (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}

		// Call Kernel for each channel
		printf("SONG = %lld\n", props1.frames);
		printf("IMP = %lld\n", props2.frames);
		printf("CV = %lld\n", props3.frames);
		printf("CUDA kernel launch with %d blocks of %d threads\n", (unsigned int)(props3.frames/THREADSPERBLOCK), THREADSPERBLOCK);
		conv_shared<<<props3.frames/THREADSPERBLOCK,THREADSPERBLOCK>>>(d_song, d_imp, d_conv, props2.frames, props3.frames);
		
		err = cudaGetLastError();
		if (err != cudaSuccess)
		{
			fprintf(stderr, "Failed to launch vectorMultiplyAdd kernel (error code %s)!\n", cudaGetErrorString(err));
			exit(EXIT_FAILURE);
		}
		
		// Copy convolved data from Device to Host
		err = cudaMemcpy(buf3[k], d_conv, props3.frames*sizeof(float), cudaMemcpyDeviceToHost);
    	if (err != cudaSuccess)
    	{
			fprintf(stderr, "Failed to copy conv vector from device to host (error code %s)!\n", cudaGetErrorString(err));
		}
	
		// Cuda device synchronize
		cudaDeviceSynchronize();
	}
#endif


	// Join output file
#ifdef CPU
	printf("channel_join CPU\n");
    channel_join(buf3, props1.channels, b3, outFrames);
#else
	printf("channel_join GPU\n");
	if(props1.channels == 1)
	{
		memcpy(b3, buf3[0], props3.frames*sizeof(float));
	}
	else if (props1.channels == 2)
	{
    	cudaMalloc((void **)&d_ch1, props3.frames*sizeof(float));
		cudaMemcpy(d_ch1, buf3[0], props3.frames*sizeof(float), cudaMemcpyHostToDevice);
    	cudaMalloc((void **)&d_ch2, props3.frames*sizeof(float));
		cudaMemcpy(d_ch2, buf3[1], props3.frames*sizeof(float), cudaMemcpyHostToDevice);
    	cudaMalloc((void **)&d_result, props1.channels*props3.frames*sizeof(float));
		channel_join_kernel<<<props3.frames/THREADSPERBLOCK,THREADSPERBLOCK>>>(d_ch1, d_ch2, d_result, props3.frames);
		cudaMemcpy(b3, d_result, props1.channels*props3.frames*sizeof(float), cudaMemcpyDeviceToHost);
	}
#endif
	printf("Convolution done!!\n");	

	// Normalize
#ifdef CPU
	printf("Normalization CPU\n");
    normalize(b3, outFrames*props1.channels, 1.5);
#else
	printf("Normalization CPU\n");
    cudaMalloc((void **)&d_b3, props1.channels*props3.frames*sizeof(float));
	cudaMemcpy(d_b3, b3, props1.channels*props3.frames*sizeof(float), cudaMemcpyHostToDevice);
	normalize_kernel<<<(props1.channels*props3.frames)/THREADSPERBLOCK,THREADSPERBLOCK>>>(d_b3, props1.channels*props3.frames, (NORMALIZE_RATIO/LOUDEST_SAMPLE));
	cudaMemcpy(b3, d_b3, props1.channels*props3.frames*sizeof(float), cudaMemcpyDeviceToHost);
#endif
    
	// Write file
    sf_writef_float(outfile, b3, outFrames);

    // Clear up
    if (infile1) sf_close(infile1);
    if (infile2) sf_close(infile2);
    if (outfile) sf_close(outfile);

    for (i = 0; i < props1.channels; i++)
    {
        if (buf1[i]) free(buf1[i]);
        if (buf2[i]) free(buf2[i]);
        if (buf3[i]) free(buf3[i]);
    }

    if (buf1) free(buf1);
    if (buf2) free(buf2);
    if (buf3) free(buf3);

    if (b1) free(b1);
    if (b2) free(b2);

    // Exit with an all clear
    end_time = time(NULL);
    process_time = difftime(end_time, current_time);
    printf("Convolution complete in %f seconds.\n", process_time);
    return 0;
}
