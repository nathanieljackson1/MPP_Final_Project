
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = emisweep
OBJ	        = main.o support.o emi_reduce_scan.o

default: $(EXE)

main.o: main.cu kernel.cu support.h emi_reduce_scan.h
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

emi_reduce_scan.o: emi_reduce_scan.cu emi_reduce_scan.h support.h
	$(NVCC) -c -o $@ emi_reduce_scan.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)
