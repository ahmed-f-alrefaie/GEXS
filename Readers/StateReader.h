#include <string>

#pragma once


class StateReader{
	protected:
		std::string m_filename;
		double partition;	
		bool open_file;	
	public: 
		StateReader(std::string pFilename="",double pPartition=-1.0);
		~StateReader();
		virtual bool OpenFile(std::string pFilename)=0;
		virtual bool CloseFile()=0;
		virtual bool ReadNextState(double & nu,double & gns,double & e_i, double & aif, double & gam,double & n)=0;
		virtual bool ReadNextState(double & nu,double & gns,double & e_i, double & aif)=0;
		virtual double ComputePartition(double temperature)=0;
};
		
		
