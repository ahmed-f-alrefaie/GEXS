



__global__ void device_compute_cross_section_voigtQ_stepone(double* g_energies,const int*  g_gns,const double*  g_nu,const double*  g_aif,double*  g_gamma,double*  g_n, const int N_ener){
	//The stored shared data
	
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double ei,gns,nu_if,aif,abscoef;
	double gammaL;
	
	//cross_constants.ln2pi/cross_constants.halfwidth;
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N_ener){
			//Store values in local memory
			ei = g_energies[g_idx];
			gns = g_gns[g_idx];
			nu_if = g_nu[g_idx];
			aif = g_aif[g_idx];

			if(nu_if==0) nu_if = 1e-6;

			abscoef= cross_constants.cmcoef*aif*gns
				*exp(-cross_constants.beta*ei)*(1.0-exp(-cross_constants.beta*nu_if))/
				(nu_if*nu_if*cross_constants.partition);

			if(gns==-1) abscoef = aif;

			g_energies[g_idx] = abscoef;

			gammaL = g_gamma[g_idx]*pow(296.0/cross_constants.temperature,g_n[g_idx])*cross_constants.pressure; 
			g_gamma[g_idx] = gammaL;

			//if(threadIdx.x == 0) printf("%14.2E   %14.2E\n",abscoef,gammaL) ;
			
	}


}
