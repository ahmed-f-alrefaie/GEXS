goal:   gpu_cross.x



PLAT = GPU_II






OBJDIR = $(HOME)/gpu_cross/obj

HEADERS := $(addprefix -I,$(shell find . -name '*.h' -printf '%h\n'))

CUDAINC := $(addprefix -I,$(shell find . -name '*.cuh' -printf '%h\n'))
INC = $(HEADERS) -I/usr/local/cuda-7.0/include $(CUDAINC)



FOR = ifort
ICC = icc
NVCC = nvcc
ICCFLAGS = -std=c++11 -traceback -O3 -xHost -openmp $(INC)
#ICCFLAGS = -std=c++11 -O0 -g $(INC)
FORFLAGS = -traceback -O3 -xHost -openmp
#ICCFLAGS = -std=c++11 -traceback -O0 -g $(INC)
#FORFLAGS = -std=c++11 -traceback -O0 -g 

#CCFLAGS = -O3 -xHost
NVCCFLAGS = --ptxas-options=-v -O3 -arch=sm_35 -Xptxas -v -lineinfo  $(INC)
#NVCCFLAGS = --ptxas-options=-v -O0 -G -g -arch=sm_35 -Xptxas -v -lineinfo  $(INC)

ICCFLAGS := $(ICCFLAGS) -DGPU_ENABLED
CUDA_OBJ = MultiGPUManager.o GPUManager.o cuda_utils.o cross_kernal_v2.o
CUDA_LIB = -L/usr/local/cuda-7.0/lib64 -lcudart -lcuda  


LIB = -lifcore -limf -lpthread $(CUDA_LIB) -openmp



###############################################################################

OBJ =  exomol_functions.o  Input.o Timer.o Util.o HITRANStateReader.o BD_TIPS_2011_v1p0.o ExomolStateReader.o BaseProfile.o BaseManager.o StateReader.o  OpenMPManager.o read_compress_trans.o bzlib.o profiles.o HybridManager.o $(CUDA_OBJ)
      # cprio.o

gpu_cross.x:       $(OBJ) main.o
	$(ICC) -o gpu_cross_$(PLAT).x $(OBJ) main.o /home/ucapfal/bz2/bzip2-1.0.6/libbz2.a $(ICCFLAGS) $(LIB) $(INC)

main.o:       main.cpp $(OBJ) 
	$(ICC) -c main.cpp $(ICCFLAGS)

cross_kernal.o: compute/CUDA/cross_kernal.cu
	$(NVCC) -c compute/CUDA/cross_kernal.cu $(NVCCFLAGS) 

cross_kernal_v2.o: compute/CUDA/cross_kernal_v2.cu
	$(NVCC) -c compute/CUDA/cross_kernal_v2.cu $(NVCCFLAGS) -I/home/ucapfal/cub/cub-1.5.1/

cuda_utils.o: compute/CUDA/cuda_utils.cu
	$(NVCC) -c compute/CUDA/cuda_utils.cu $(NVCCFLAGS) 

HITRANStateReader.o: Readers/HITRAN/HITRANStateReader.cpp
	$(ICC) -c Readers/HITRAN/HITRANStateReader.cpp $(ICCFLAGS) $(INC)

ExomolStateReader.o: Readers/Exomol/ExomolStateReader.cpp
	$(ICC) -c Readers/Exomol/ExomolStateReader.cpp $(ICCFLAGS)

StateReader.o: Readers/StateReader.cpp
	$(ICC) -c Readers/StateReader.cpp $(ICCFLAGS)

BaseProfile.o: Profiles/BaseProfile.cpp
	$(ICC) -c Profiles/BaseProfile.cpp $(ICCFLAGS)

VoigtProfile.o: Profiles/Voigt/VoigtProfile.cpp
	$(ICC) -c Profiles/Voigt/VoigtProfile.cpp $(ICCFLAGS)

DopplerProfile.o: Profiles/Doppler/DopplerProfile.cpp
	$(ICC) -c Profiles/Doppler/DopplerProfile.cpp $(ICCFLAGS)

GaussianProfile.o: Profiles/Gaussian/GaussianProfile.cpp
	$(ICC) -c Profiles/Gaussian/GaussianProfile.cpp $(ICCFLAGS)

GPUManager.o: Managers/Accel/GPU/GPUManager.cpp
	$(NVCC) -c Managers/Accel/GPU/GPUManager.cpp $(NVCCFLAGS) 

HybridManager.o: Managers/Hybrid/HybridManager.cpp
	$(ICC) -c Managers/Hybrid/HybridManager.cpp $(ICCFLAGS)

BaseManager.o: Managers/BaseManager.cpp
	$(ICC) -c Managers/BaseManager.cpp $(ICCFLAGS)

BD_TIPS_2011_v1p0.o: 
	$(FOR) -c Readers/HITRAN/BD_TIPS_2011_v1p0.for $(FORFLAGS)

MultiGPUManager.o: Managers/Accel/GPU/MultiGPUManager.cpp
	$(ICC) -c Managers/Accel/GPU/MultiGPUManager.cpp $(ICCFLAGS)
	
Input.o: common/Input.cpp
	$(ICC) -c common/Input.cpp $(ICCFLAGS)

Util.o: common/Util.cpp
	$(ICC) -c common/Util.cpp $(ICCFLAGS)

Timer.o: common/Timer.cpp   
	$(ICC) -c common/Timer.cpp $(ICCFLAGS)

OpenMPManager.o: Managers/CPU/OpenMPManager.cpp
	$(ICC) -c Managers/CPU/OpenMPManager.cpp $(ICCFLAGS)

profiles.o: compute/CPU/profiles.cpp
	$(ICC) -c compute/CPU/profiles.cpp $(ICCFLAGS) 

exomol_functions.o: Readers/Exomol/exomol_functions.cpp
	$(ICC) -c Readers/Exomol/exomol_functions.cpp $(ICCFLAGS)

bzlib.o: Readers/Exomol/bz2_compression/bzlib.c
	$(ICC) -c Readers/Exomol/bz2_compression/bzlib.c  $(ICCFLAGS) 

read_compress_trans.o: Readers/Exomol/bz2_compression/read_compress_trans.c
	$(ICC) -c Readers/Exomol/bz2_compression/read_compress_trans.c $(ICCFLAGS) 
clean:
	rm $(OBJ) *.o

