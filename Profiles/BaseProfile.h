//##include "MultiGPUManager.h"
#include "BaseManager.h"
#include "StateReader.h"
#include "HITRANStateReader.h"
#include "ExomolStateReader.h"
#include "Input.h"
#include "Timer.h"
#include "HybridManager.h"
#pragma once


class BaseProfile{

	private:
		BaseManager* manager;
		StateReader* state_reader;
		Input* input;
		ProfileType profile;
		double* freq;
		double* intens;
		double dfreq;
		int num_intens;
		double partition;

		double total_intens;

		int ib,ie;
		double* h_energies; 
		double* h_nu;
		double * h_aif;
		int* h_gns;
		double* h_gammaL;
		double* h_n;
		
		double m_beta;		

		int Npoints;
		int num_trans_fit;
		int Ntrans;
		int points;
		int old_Ntrans;
		double start_nu,end_nu,min_nu,max_nu,max_hw;
		vector<string> transition_files;
		vector<double> temperature_comp;
		vector<TemperaturePressure> temperature_pressure_comp;
		void ComputeTotalIntensity();
	protected:
		double m_half_width;
		BaseProfile(Input* pInput);
		~BaseProfile();
		bool require_pressure;
		void PerformCrossSection(double HW);
		virtual ProfileType GetProfile()=0;
		void ComputePartitions();
	public:
		void Initialize();
		void ExecuteCrossSection();
		void OutputProfile();
};
