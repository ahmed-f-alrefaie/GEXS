#include "BaseProfile.h"
#include "HITRANStateReader.h"
#include "ExomolStateReader.h"
#include <cmath>
#pragma once

class DopplerProfile : public BaseProfile{	
	private:

	public:
		DopplerProfile(Input* pInput) :BaseProfile(pInput){require_pressure=false; m_half_width=100.0;};
		ProfileType GetProfile(){printf("Profile Selected is Doppler\n");return DOPPLER;};


};


