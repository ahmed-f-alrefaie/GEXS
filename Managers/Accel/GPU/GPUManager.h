#ifndef GPU_MANAGER_H
#define GPU_MANAGER_H
#include "BaseManager.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>
#include "cross_kernal_v2.cuh"
#include "Input.h"
#include <vector>


class GpuManager : public BaseManager{
	private:
		int gpu_id;
		double* g_freq;
		double* g_intens;
		std::vector<double*> g_tp_intens;
		double* g_energies;
		double* g_nu;
		double * g_aif;
		std::vector<double*> g_tp_abscoef;
		double * g_gamma;
		double * g_n;
		int* g_gns;
		int tasks;
		double max_nu;
		double dpwcoeff;
		std::vector<cudaStream_t> g_tp_stream;
		//GPU pointers
		bool alloc;
		void ExecuteVoigtCrossSection(int N, int N_ener,int start_idx);
		void ExecuteVoigtCrossSectionBlock(int N, int N_ener,int start_idx);
		void ExecuteVoigtCrossSectionBlock(int N, int N_ener,int start_idx,double temperature,double pressure,double partition);
		void ExecuteGaussianCrossSection(int N, int N_ener,int start_idx);	
		void ExecuteGaussianCrossSection(int N, int N_ener,int start_idx,double temperature,double partition,double hw);
		void ExecuteDopplerCrossSection(int N, int N_ener,int start_idx);
		void ExecuteDopplerCrossSection(int N, int N_ener,int start_idx,double temperature,double partition);	

	public:
		GpuManager(ProfileType pProfile,int gpu_id=0);
		~GpuManager();
		void InitializeVectors(int Npoints);
		void InitializeVectors(int Npoints,int num_cs);
		void InitializeConstants(double half_width,double temperature, double partition,double dfreq,double meanmass,double pressure=10.0,double ref_temp=296.0);
		void TransferVectors(size_t Nener,double* h_energies, double* h_nu, double* h_aif,int* h_gns,double* h_gamma=NULL,double * h_n=NULL);
		void TransferFreq(double* h_freq,double* h_intens,int N);
		void ExecuteCrossSection(int N, int N_ener,int start_idx);
		void ExecuteCrossSection(int N, int N_ener,int start_idx,double temperature,double pressure,double partition);
		void TransferResults(double* h_freq,double* h_intens,int N);
		void TransferResults(double* h_freq,double* h_intens,int N,int I);
		void Cleanup();
		bool ReadyForWork();
		

};

#endif
