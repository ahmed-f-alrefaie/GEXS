#include "BaseProfile.h"
#include "HITRANStateReader.h"
#include "ExomolStateReader.h"

#include <cmath>
#pragma once

class VoigtQuadProfile : public BaseProfile{	
	private:
		//The double and	
		double max_gammaL;
		double min_n;
		double pressure;
		double* abciss;
		

	public:
		VoigtProfile(Input* pInput) : BaseProfile(pInput){};
		void InitializeProfile();
		void ExecuteCrossSection();


};


