# 
# Build Nueral Network tester from NVCC/GCC and libraries CUDNN/CUBLAS/CUDA
#
#
#============================================================================


# Location of the CUDA/CUDNN Toolkit
CUDA_PATH      ?= /usr/local/cuda
# HDF5_PATH 	?= /server-home1/ndanande/local/hdf5
CUDNN_PATH	?= /server-home1/ndanande/local/cuda

# architecture
HOST_ARCH      := $(shell uname -m)
TARGET_ARCH    ?= $(HOST_ARCH)


# compiler
HOST_COMPILER  ?= g++
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# flags
CCFLAGS   := -std=c++11 -O3 -g
NVCCFLAGS := -Wno-deprecated-gpu-targets -lineinfo
LDFLAGS   := -lcublas -lhdf5 -lhdf5_cpp -lcudnn

LDFLAG2 	:= -lhdf5 -lhdf5_cpp -lcudnn

# include paths
INCLUDES  += -I$(CUDA_PATH)/include
LIBRARIES += -L$(CUDA_PATH)/lib64

INCLUDES += -I$(HDF5_PATH)/include
LIBRARIES += -L$(HDF5_PATH)/lib

INCLUDES += -I$(CUDNN_PATH)/include
LIBRARIES += -L$(CUDNN_PATH)/lib64

#=====================================================================
#
# Target rules
EXE = gfn
OBJ = driver.o Gfunction_v2.o timestamps.o utility.o utility_cu.o


all: clean build

build: $(EXE)

$(EXE): $(OBJ)
	$(NVCC) -o $@ $? $(CCFLAGS) $(LIBRARIES) $(NVCCFLAGS)  $(LDFLAGS) 

%.o: %.cu
	$(NVCC) $(INCLUDES) $(LIBRARIES) $(NVCCFLAGS) $(CCFLAGS) $(LDFLAGS) $(LDFLAG2) -o $@ -c $<
	
%.o: %.cpp 
#	$(HOST_COMPILER) $(CCFLAGS) -o $@ -c $<
	$(NVCC) $(INCLUDES) $(LIBRARIES) $(NVCCFLAGS) $(CCFLAGS) $(LDFLAGS) $(LDFLAG2) -o $@ -c $<
				
clean:
	rm -rf *.o
	rm -f gfn
	
