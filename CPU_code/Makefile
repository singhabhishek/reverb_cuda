CPU_HOME := /usr/local
INC	:= -I$(CPU_HOME)/include -I.
LIB	:= -L$(CPU_HOME)/lib -lsndfile -lm 
CC	:= gcc 

CCFLAGS	:= -g

reverb_effect: reverb_effect.c
	$(CC) $(INC) $(CCFLAGS) reverb_effect.c $(LIB) -o reverb_effect

clean:
	rm -f *.o reverb_effect


