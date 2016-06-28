#include "BaseProfile.h"
#include "HITRANStateReader.h"
#include "ExomolStateReader.h"
#include <cmath>
#pragma once

class GaussianProfile : public BaseProfile{	
	private:
		//The double and	

	public:
		GaussianProfile(Input* pInput) : BaseProfile(pInput){require_pressure=false; m_half_width=10.0;};
		ProfileType GetProfile(){printf("Profile Selected is Gaussian\n");return GAUSSIAN;};


};


