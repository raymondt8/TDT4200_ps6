heat_cuda : heat_cuda.cu
	nvcc -ccbin=g++-4.8 --gpu-architecture sm_35 -O3 heat_cuda.cu -o heat_cuda

clean:
	-rm -f heat_cuda
	-rm -f data/*
