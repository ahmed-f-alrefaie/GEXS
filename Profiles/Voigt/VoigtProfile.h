#include "BaseProfile.h"


#include <cmath>
#pragma once

class VoigtProfile : public BaseProfile{	
	private:
		//The double and	
		double max_gammaL;
		double min_n;
		double pressure;

	public:
		VoigtProfile(Input* pInput,double pressure) : BaseProfile(pInput){require_pressure=true; m_half_width=LORENTZ_CUTOFF;};
		ProfileType GetProfile(){printf("Profile Selected is Voigt\n"); return VOIGT;};
		//void InitializeProfile();
		//void ExecuteCrossSection();


};


