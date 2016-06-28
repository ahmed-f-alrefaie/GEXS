#include <StateReader.h>
#include "HITRAN_func.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <cstdlib>
#pragma once

class HITRANStateReader : public StateReader{
	private:
		FILE* hitran_file;
		char buffer[1024];
		double m_pressure;
		double m_mixture;
		double m_temperature;
		int mol_id;
		int iso_num;
		double ref_partition;
	public:
		HITRANStateReader(std::string pFilename="",double partition=-1.0,double pressure=1.0,double temperature=296.0,double hitran_air_mixture=1.0);
		~HITRANStateReader();
		bool OpenFile(std::string pFilename);
		bool CloseFile();
		bool ReadNextState(double & nu,double & gns,double & e_i, double & aif, double & gam,double & n);
		bool ReadNextState(double & nu,double & gns,double & e_i, double & aif);
		double ComputePartition(double temperature);//{return partition;};
};
