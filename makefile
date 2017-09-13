ICPC = icpc
NVCC = nvcc
CC = g++
CFLAGS = -I ./ -std=c++11 -fopenmp -xhost -g
NVCCFLAGS = -I ./ -ccbin g++ -Xcompiler -fopenmp -g -D USE_CUDA -m64 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60
ICFLAGS = -I ./ -std=c++11 -qopenmp -D USE_MKL -mkl=sequential -xhost -g


SHARED := src/buffer.cpp src/console.cpp \
	src/ti_producer.cpp src/word_source_file.cpp \
	src/word_source_group.cpp src/word_source_file_group.cpp src/sentence_producer.cpp \
	src/pht_model.cpp src/sen_buffer.cpp \
	src/sgd_trainer.cpp src/tc_buffer.cpp src/unlocked_buffer.cpp \
	src/w2v-functions.cpp src/worker_model.cpp src/consumer.cpp \
	src/batch_consumer.cpp src/batch_model.cpp src/sgd_batch_trainer.cpp

all : $(SHARED) src/main.cpp 
	$(CC) $? $(CFLAGS) -O3 -o wombat 

intel : $(SHARED) src/main.cpp src/sgd_mkl_trainer.cpp
	$(ICPC) $? $(ICFLAGS) -O3 -o wombat

cuda : $(SHARED) src/main-cuda.cpp src/cuda_batch_model.cpp src/sgd_cuda_trainer.cu src/sgd_cuda_trainer.cu kernels.o
	$(NVCC) $? -o wombat $(NVCCFLAGS) -std=c++11 -O3

kernels.o:src/cuda_kernel_wrapper.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<


