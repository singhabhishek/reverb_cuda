CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib -lsndfile -lm 
CC	:= nvcc

#NVCCFLAGS	:= -lineinfo -arch=sm_53 -g
NVCCFLAGS1	:= -lineinfo -g -D WITH_STREAM
NVCCFLAGS2	:= -lineinfo -g -D WITHOUT_STREAM

all: reverb_effect_with_stream reverb_effect_without_stream

reverb_effect_with_stream: reverb_effect.cu
	$(CC) $(INC) $(NVCCFLAGS1) reverb_effect.cu $(LIB) -o reverb_effect_with_stream

reverb_effect_without_stream: reverb_effect.cu
	$(CC) $(INC) $(NVCCFLAGS2) reverb_effect.cu $(LIB) -o reverb_effect_without_stream


clean:
	rm -f *.o reverb_effect_*


