#include <cmath>
#include <thread>
#include "BaseManager.h"
#include <vector>
#include "defines.h"
#include "Input.h"
#include "SafeQueue.h"
#pragma once

struct ThreadJob{
	int N;
	int N_ener;
	int start_idx;
	double temperature;
	double pressure;
	double partition;
};
//This is a lie it is not openmp but uses threads
class OpenMPManager : public BaseManager{

private:
		bool* threads_done;
		bool master_thread_done;
		int total_threads;
		std::vector<std::thread> workers;
		std::thread* master_thread;
		double** t_intens; 
		double m_half_width;
		double m_beta;
		double m_partition;
		double m_ref_temp;
		double m_dfreq;
		double m_temperature;
		double m_mean_mass;
		double m_pressure;	
		double m_cmcoef;
		double* g_freq;
		double* g_intens;
		double* g_energies;
		double* g_nu;
		double * g_aif;
		double * g_gamma;
		double * g_n;
		int* g_gns;
		int tasks;
		std::vector<TemperaturePressure> temp_press_queue;
		std::vector<double*> g_tp_intens;
		std::mutex g_intens_mutex;	
		SafeQueue<ThreadJob> thread_jobs;
		double start_nu;	
		void DistributeWork();
		void JoinAllThreads();
		void ComputeVoigt(int id,int Npoints,int Nener,int start_idx);	
		void ComputeVoigtTP(int id,int Npoints,int Nener,int start_idx,int tasks,double temperature, double pressure,double partition);
		void ComputeDoppler(int id,int Npoints,int Nener,int start_idx);
		//bool ReadyForThreadWork();		

public:
		OpenMPManager(ProfileType pprofile, int num_threads,size_t total_memory);
		void InitializeVectors(int Npoints);
		void InitializeVectors(int Npoints,int num_cs);
		void InitializeConstants(double half_width,double temperature, double partition,double dfreq,double meanmass,double pressure=10.0,double ref_temp=296.0,double ref_press=1.0);
		void TransferVectors(size_t Nener,double* h_energies, double* h_nu, double* h_aif,int* h_gns,double* h_gamma=NULL,double * h_n=NULL);
		void TransferFreq(double* h_freq,double* h_intens,int N);
		void ExecuteCrossSection(int N, int N_ener,int start_idx);
		void ExecuteCrossSection(int N, int N_ener,int start_idx,double temperature,double pressure,double partition);
		void TransferResults(double* h_freq,double* h_intens,int N);
		void TransferResults(double* h_freq,double* h_intens,int N,int I);
		void Cleanup();
		bool ReadyForWork();

		

};
