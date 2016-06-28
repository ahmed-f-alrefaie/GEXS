#include <cmath>
#include <cstdlib>
#include "Input.h"
#pragma once

class BaseManager{
private:
		size_t available_memory;
		size_t total_memory;
protected:
		size_t N_trans;

		ProfileType profile;
		void InitializeMemory(size_t bytes);
		void TrackMemory(size_t bytes);
		void FreeMemory(size_t bytes);
		size_t GetAvailableMemory();
		BaseManager(ProfileType pprofile);
		~BaseManager();
		
public:
		virtual void InitializeVectors(int Npoints)=0;
		virtual void InitializeVectors(int Npoints,int num_cs)=0;
		//virtual void InitializeVectors(int Npoints,int N_intens)=0;
		virtual void InitializeConstants(double half_width,double temperature, double partition,double dfreq,double meanmass,double pressure=10.0,double ref_temp=296.0)=0;
		virtual void TransferVectors(size_t Nener,double* h_energies, double* h_nu, double* h_aif,int* h_gns,double* h_gamma=NULL,double * h_n=NULL)=0;
		virtual void TransferFreq(double* h_freq,double* h_intens,int N)=0;
		virtual void ExecuteCrossSection(int N, int N_ener,int start_idx)=0;
		//virtual void ExecuteCrossSection(int N, int N_ener,int start_idx,double temperature,double partition);
		virtual void ExecuteCrossSection(int N, int N_ener,int start_idx,double temperature,double pressure,double partition)=0;
		virtual void TransferResults(double* h_freq,double* h_intens,int N)=0;
		virtual void TransferResults(double* h_freq,double* h_intens,int N,int I)=0;
		virtual void Cleanup()=0;
		virtual bool ReadyForWork();
		size_t GetNtrans();
		

};

