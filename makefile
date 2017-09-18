ICPC = icpc
NVCC = nvcc
CC = g++
CFLAGS = -I ./ -std=c++11 -fopenmp -xhost -g
NVCCFLAGS = -I ./ -ccbin g++ -Xcompiler -fopenmp -g -D USE_CUDA -m64 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_60,code=compute_60
ICFLAGS = -I ./ -std=c++11 -qopenmp -D USE_MKL -mkl=sequential -xhost -g


SHARED := src/console.cpp \
	src/buffers/producers/ti_producer.cpp src/data_source/word_source_file.cpp \
	src/data_source/word_source_group.cpp src/data_source/word_source_file_group.cpp src/buffers/producers/sentence_producer.cpp \
	src/pht_model.cpp src/buffers/sen_buffer.cpp src/buffers/readers/sen_buffer.cpp \
	src/sgd_trainers/sgd_trainer.cpp src/buffers/tc_buffer.cpp src/buffers/readers/tc_buffer.cpp  \
	src/w2v-functions.cpp src/worker_model.cpp src/consumer.cpp \
	src/batch_consumer.cpp src/batch_model.cpp src/sgd_trainers/sgd_batch_trainer.cpp

all : $(SHARED) src/main.cpp 
	$(CC) $? $(CFLAGS) -O3 -o wombat 

intel : $(SHARED) src/main.cpp src/sgd_trainers/sgd_mkl_trainer.cpp
	$(ICPC) $? $(ICFLAGS) -O3 -o wombat

cuda : $(SHARED) src/main-cuda.cpp src/cuda_batch_model.cpp src/sgd_trainers/sgd_cuda_trainer.cu src/sgd_trainers/sgd_cuda_trainer.cu kernels.o
	$(NVCC) $? -o wombat $(NVCCFLAGS) -std=c++11 -O3

kernels.o:src/sgd_trainers/cuda_kernel_wrapper.cu
	$(NVCC) $(NVCCFLAGS) -o $@ -c $<


