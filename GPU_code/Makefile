CUDA_HOME := /usr/local/cuda
INC	:= -I$(CUDA_HOME)/include -I.
LIB	:= -L$(CUDA_HOME)/lib -lsndfile -lm 
CC	:= nvcc

#NVCCFLAGS	:= -lineinfo -arch=sm_53 -g
NVCCFLAGS	:= -lineinfo -g

reverb_effect: reverb_effect.cu
	$(CC) $(INC) $(NVCCFLAGS) reverb_effect.cu $(LIB) -o reverb_effect

clean:
	rm -f *.o reverb_effect


