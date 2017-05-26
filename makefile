ICPC = icpc
NVCC = nvcc
CC = g++
CFLAGS = -std=c++11 -fopenmp -xhost -g
NVCCFLAGS = -ccbin g++ -Xcompiler -fopenmp -g -D USE_CUDA -m64 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60
ICFLAGS = -std=c++11 -qopenmp -D USE_MKL -mkl=sequential -xhost -g


SHARED := src/buffer.cpp src/console.cpp src/shared_consumer.cpp \
	src/ti_producer.cpp src/word_source_file.cpp \
	src/word_source_group.cpp src/word_source_file_group.cpp src/sentence_producer.cpp \
	src/pht_model.cpp src/sen_buffer.cpp \
	src/sgd_trainer.cpp src/tc_buffer.cpp src/timer.cpp src/unlocked_buffer.cpp \
	src/w2v-functions.cpp src/worker_model.cpp src/pht_nested_model.cpp src/consumer.cpp \
	src/batch_consumer.cpp src/batch_model.cpp src/sgd_batch_trainer.cpp

all : $(SHARED) main.cpp 
	$(CC) $? $(CFLAGS) -O3 -o wombat 

intel : $(SHARED) main.cpp src/sgd_mkl_trainer.cpp
	$(ICPC) $? $(ICFLAGS) -O3 -o wombat

cuda : $(SHARED) main-cuda.cpp src/cuda_batch_model.cpp src/sgd_cuda_trainer.cu src/sgd_cuda_trainer.cu kernels.o
	$(NVCC) $? -o wombat $(NVCCFLAGS) -std=c++11 -O3

kernels.o:src/cuda_kernel_wrapper.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<


