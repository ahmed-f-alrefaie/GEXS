#ifndef MGPU_MANAGER_H
#define MGPU_MANAGER_H
#include "BaseManager.h"
#include "GPUManager.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>
#include "cross_kernal.cuh"
#include "Input.h"



class MultiGpuManager : public BaseManager{
	private:
		int t_num_gpus;
		int selected_gpu;
		void SwitchGPU();
		GpuManager** gpu_managers;
		GpuManager* selected_manager;

	public:
		MultiGpuManager(ProfileType pProfile);
		~MultiGpuManager();
		void InitializeVectors(int Npoints);
		void InitializeConstants(double half_width,double temperature, double partition,double dfreq,double meanmass,double pressure=10.0,double ref_temp=296.0);
		void TransferVectors(size_t Nener,double* h_energies, double* h_nu, double* h_aif,int* h_gns,double* h_gamma=NULL,double * h_n=NULL);
		void TransferFreq(double* h_freq,double* h_intens,int N);
		void ExecuteCrossSection(int N, int N_ener,int start_idx);
		void TransferResults(double* h_freq,double* h_intens,int N);
		void Cleanup();

		

};

#endif
