#include "BaseManager.h"
#include "OpenMPManager.h"
#include <cmath>
#include <vector>
#include "Input.h"
#ifdef GPU_ENABLED
#include "MultiGPUManager.h"
#endif

#pragma once

class HybridManager : public BaseManager{
	private:
		int t_num_workers;
		int selected_worker;
		int num_intens;
		std::vector<BaseManager*> workers;
		BaseManager* current_worker;
		void SelectWorker();

	public:
		HybridManager(ProfileType pProfile,int num_threads,size_t total_memory);
		~HybridManager();
		void InitializeVectors(int Npoints);
		void InitializeVectors(int Npoints,int num_cs);
		void InitializeConstants(double half_width,double temperature, double partition,double dfreq,double meanmass,double pressure=10.0,double ref_temp=296.0);
		void TransferVectors(size_t Nener,double* h_energies, double* h_nu, double* h_aif,int* h_gns,double* h_gamma=NULL,double * h_n=NULL);
		void TransferFreq(double* h_freq,double* h_intens,int N);
		void ExecuteCrossSection(int N, int N_ener,int start_idx);
		void ExecuteCrossSection(int N, int N_ener,int start_idx,double temperature,double pressure,double partition);
		void TransferResults(double* h_freq,double* h_intens,int N);
		void TransferResults(double* h_freq,double* h_intens,int N,int I);
		bool ReadyForWork();
		void Cleanup();
		

		

};

