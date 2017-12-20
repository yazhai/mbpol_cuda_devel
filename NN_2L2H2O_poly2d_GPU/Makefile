# 
# Build Nueral Network tester from NVCC/GCC and libraries CUDNN/CUBLAS/CUDA
#
#
#============================================================================


# Location of the CUDA/CUDNN Toolkit
CUDA_PATH      ?= /usr/local/cuda

# architecture
HOST_ARCH      := $(shell uname -m)
TARGET_ARCH    ?= $(HOST_ARCH)


# compiler
HOST_COMPILER  ?= g++
NVCC           := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)

# flags
CCFLAGS   := -std=c++11 -O3
NVCCFLAGS := -Wno-deprecated-gpu-targets  
LDFLAGS   := -lcublas


# include paths
INCLUDES  += -I$(CUDA_PATH)/include
LIBRARIES += -L$(CUDA_PATH)/lib64



#=====================================================================
#
# Target rules
EXE = gfn
OBJ = atomTypeID.o Gfunction_cu.o readGparams.o timestamps.o utility.o utility_cu.o


all: clean build

build: $(EXE)


$(EXE): $(OBJ)
	$(NVCC) -o $@ $(OBJ) $(CCFLAGS) $(NVCCFLAGS)  $(LDFLAGS) 

%.o: %.cu 
	$(NVCC) $(INCLUDES) $(LIBRARIES) $(NVCCFLAGS) $(CCFLAGS) $(LDFLAGS) -o $@ -c $<
	
%.o: %.cpp 
	$(HOST_COMPILER) $(CCFLAGS) -o $@ -c $<
				
clean:
	rm -rf *.o
	rm -f gfn
	
