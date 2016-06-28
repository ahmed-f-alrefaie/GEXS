#include "cross_structs.cuh"
#include "cross_kernal.cuh"
#include "cuda_utils.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include <cmath>
#include <cuComplex.h>
#include <thrust/complex.h>

#include "defines.h"

__constant__ cross_section_data cross_constants;
__constant__ double weights[] = {1.43956E-6,3.46819E-4,0.0119114,0.117228,0.42936};
__constant__ double abscissa[] ={13.457672141,7.746703224,4.104473403,1.759761434,0.431399376};  	
//Lean and mean version of the sigma
__device__ void Sigmalean(double x, double y, double &s1, double &s2, double &s3, double a){
	//sigma1	
	s1 = 0.0;
	//sigma2
	s2 = 0.0;
	//sigma3
	s3 = 0.0;
	
	//Precompute some values here
	double ex2 = exp(-x*x);
	double an,a2n2,f,ea2n2,e2anx,e2anx2;
	for(int n=1 ; n < 13; n++){
		an = a*double(n);
		a2n2 = an*an;
		f = 1.0/(a2n2 + y*y);		
		//ea2n2 = exp(-a2n2);
		e2anx = (an+x);
		e2anx2 = (an-x);		
		//e2anx2 = exp(2.0*an*x);	
		s1 += f*exp(-(a2n2+x*x));//ea2n2*ex2;
		s2 += f*exp(-e2anx*e2anx);//ea2n2*e2anx*ex2;
		s3 += f*exp(-e2anx2*e2anx2);//ea2n2*e2anx2*ex2;
	}
	//s1*=ex2;
	//s2*=1.0f;//ex2;
	//s3*=1.0f;//*=ex2;
	

}






// *********************************************
//This function calculates the Series Sigma1. Sigma2 and Sigma3 (Equations 15, 16, and 17) from Alg 916
//The parameter TOL sets a tolerance where to truncate the series
//It returns the values for sigma1 sigma2 and sigma3

//Author Simon Grimm, Adapted from Zaghloul & Ali, Algorithm 916
//November 2014
// **********************************************

__device__ void Sigmab(double x, double y, double &s1, double &s2, double &s3, double a){

	s1 = 0.0;
	//double sold1 = s1;
	s2 = 0.0;
	//double sold2 = s2;
	s3 = 0.0;
	//double sold3 = s3;

	double f, f3p, f3n;
	double an, an3p, an3n;

	double yy = y * y;

	//if(x < 0.0) x = -x;

	int n0 = (int)(ceil(x / a)); //starting point for sigma3 series
	int n3p, n3n;

	//int stop1 = 0;
	//int stop2 = 0;
	//int stop3 = 0;

	//double e2axn = exp(-2.0 * a * x);
	for(int n = 1; n < 5; ++n){
		n3p = n0 + n - 1;
		n3n = n0 - n;
		an = a * n;
		an3p = a * n3p;
		an3n = a * n3n;

		f = 1.0 / (an * an + yy);
		f3p = 1.0 / (an3p * an3p + yy);
		f3n = 1.0 / (an3n * an3n + yy);

		s1 += f * exp(-(an * an + x * x));
		s2 += f * exp(-(an + x) * (an + x));
		s3 += f3p * exp(-(an3p - x) * (an3p - x));
		if(n3n >= 1) s3 += f3n * exp(-(an3n - x) * (an3n - x));

		//if(fabs(s1 - sold1) < TOL) stop1 = 1;
		//if(fabs(s2 - sold2) < TOL) stop2 = 1;
		//if(fabs(s3 - sold3) < TOL) stop3 = 1;
		//if(stop1 == 1 && stop2 ==1 && stop3 == 1) break;

		//sold1 = s1;
		//sold2 = s2;
		//sold3 = s3;
//		if(n >= 100-1) printf("Sigma Series did not converge %d\n", id);
	}
};

// *************************************************
//This function calculates the Voigt profile V(x,y) as equation 13 from Zaghloul & Ali, Algorithm 916
//it calls the Sigma function
//The parameter TOL sets a tolerance where to truncate the series

//Author Simon Grimm, Adapted from Zaghloul & Ali, Algorithm 916
//November 2014
// **************************************************
__device__ double voigt_916(double x, double y, double a){

	double s1, s2, s3;
	double ex2 = exp(-x * x);


	if(x == 0) return erfcx(y);
	if(y == 0) return ex2;	
	//Compute Sigma Series
	if(x != 0.0 && y != 0.0) Sigmab(x, y, s1, s2, s3, a);



	double xy = x * y;
	double a2ipi = 2.0 * a / M_PI;
	double cos2xy = cos(2.0 * xy);
	double sinxy = sin(xy);

	double t1 = ex2 * erfcx(y) * cos2xy;
	t1 += a2ipi * x * sinxy * ex2 * sinxy / xy;
	t1 += a2ipi * y * (-cos2xy * s1 + 0.5 * (s2 + s3));
	

	//if(x*x + y*y > 1.0e18) t1 = y / (sqrt(M_PI) * (x * x + y * y));

	return t1;
};

__device__ double humlick(double x, double y){
	thrust::complex<double> T = thrust::complex<double>(y, -x);
	thrust::complex<double> humlic1;
   // double complex T = y - x*I;
    double S = fabs(x) + y;
    if (S >= 15) {
        // Region I
        humlic1 = T*0.5641896/(0.5+T*T);
        //fprintf(stdout, "I");
     
    }else if (S >= 5.5) {
        // Region II
        thrust::complex<double> U = T * T;
        humlic1 = T * (1.410474 + U*.5641896)/(.75 + U*(3.+U));
        //fprintf(stdout, "II");
    }else if (y >= 0.195 * fabs(x) - 0.176) {
        // Region III
        humlic1 = (16.4955+T*(20.20933+T*(11.96482
                +T*(3.778987+T*.5642236)))) / (16.4955+T*(38.82363
                +T*(39.27121+T*(21.69274+T*(6.699398+T)))));
        //fprintf(stdout, "III");
    }else{
    // Region IV
    thrust::complex<double> U = T * T;
    //double complex humlic1;
    humlic1 = thrust::exp(U)-T*(36183.31-U*(3321.9905-U*(1540.787-U*(219.0313-U*
       (35.76683-U*(1.320522-U*.56419))))))/(32066.6-U*(24322.84-U*
       (9022.228-U*(2186.181-U*(364.2191-U*(61.57037-U*(1.841439-U)))))));
    //fprintf(stdout, "IV");
    }    
    return humlic1.real();

};

__device__ double voigt_threegausshermite(double x, double y,double xxyy){
	
	return 1.1181635900*y*IPI/(xxyy) + 2.0*IPI*y*(xxyy + 1.499988068)/( (xxyy+1.499988068)*(xxyy+1.499988068) - 4*x*x*1.499988068);
};

__device__ double voigt_elevengausshermite(double & x, double & y,double & xxyy){
	/*
	double voigt_=0.0;
	
	for(int i = 0; i < 5; i++){
		double ab=abscissa[i];
		voigt_ +=weights[i]*y*(xxyy+ab)/((xxyy+ab)*(xxyy+ab)-4.0*x*x*ab); 
	}*/
	
	double f1,f2,f3,f4,f5,f6;
	double xxyyxxyy = xxyy*xxyy;
	double fourxx = 4.0*x*x;
	f6 = 1.43956E-6/xxyy;
	f1 = 3.46819E-4*(xxyy + 13.457672141)/(xxyyxxyy + 181.108939455 + 2.0*xxyy*13.457672141 - 4.0*x*x*13.457672141);

	f2 = 0.0119114*(xxyy + 7.746703224)/(xxyyxxyy + 60.011410842 + 2.0*xxyy*7.746703224 - fourxx*7.746703224);


	f3 = 0.117228*(xxyy + 4.104473403)/(xxyyxxyy + 16.846701914 + 2.0*xxyy*4.104473403 - fourxx*4.104473403);
 	

        f4 = 0.42936*(xxyy + 1.759761434)/(xxyyxxyy + 3.096760304 + 2.0*xxyy*1.759761434 - fourxx*1.759761434);


        f5 = 0.654759*(xxyy + 0.431399376)/(xxyyxxyy + 0.186105422 + 2.0*xxyy*0.431399376 - fourxx*0.431399376);
 	

	return (f6+2.0*(f1+f2+f3+f4+f5))*y/PI;
};

__host__ void copy_intensity_info(cross_section_data* cross_inf)
{
	//void* ptr;
	//cudaGetSymbolAddress ( &ptr, int_info );

	cudaMemcpyToSymbol(cross_constants, (void*)cross_inf, sizeof(cross_section_data),0,cudaMemcpyHostToDevice);
};

__host__ void execute_two_step_kernal(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif, int Npoints,int N_ener){
		
		int blockSize = 1024;
		int gridSize = (int)ceil((float)Npoints/blockSize);
		device_compute_cross_section_stepone<<<gridSize,blockSize>>>(g_energies,g_gns,g_nu,g_aif,N_ener);
		blockSize = BLOCK_SIZE;
		gridSize = (int)ceil((float)Npoints/blockSize);

		device_compute_cross_section_steptwo<<<gridSize,blockSize>>>(g_freq, g_intens,g_nu,g_energies,Npoints,N_ener,0);

					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
}

__host__ void execute_two_step_kernal_block(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif, int Npoints,int N_ener,int start_idx){
		
		int blockSize = 1024;
		int gridSize = (int)ceil((float)N_ener/blockSize);
		device_compute_cross_section_doppler_stepone<<<gridSize,blockSize>>>(g_energies,g_gns,g_nu,g_aif,N_ener);
		blockSize = BLOCK_SIZE;
		gridSize = Npoints;//(int)ceil((float)Npoints/blockSize);
		//printf("gridSize = %d, blockSize = %d\n",gridSize,blockSize);
		 cudaFuncSetCacheConfig(device_compute_cross_section_steptwo_block, cudaFuncCachePreferL1);
		device_compute_cross_section_steptwo_block<<<gridSize,blockSize>>>(g_freq, g_intens,g_nu,g_energies,Npoints,N_ener,start_idx);
		//cudaDeviceSynchronize();
		//CheckCudaError("Step Two");
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
}

__host__ void execute_two_step_kernal_block_temp(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double temp,double part,double hw, int Npoints,int N_ener,int start_idx,cudaStream_t stream){
		
		int blockSize = 1024;
		int gridSize = (int)ceil((float)N_ener/blockSize);
		device_compute_cross_section_doppler_stepone_temp<<<gridSize,blockSize,0,stream>>>(g_energies,g_gns,g_nu,g_aif,g_abs,temp,part,N_ener);
		blockSize = SHARED_SIZE;
		gridSize = (int)ceil((float)Npoints/blockSize);
		printf("gridSize = %d, blockSize = %d\n",gridSize,blockSize);
		// cudaFuncSetCacheConfig(device_compute_cross_section_steptwo_block, cudaFuncCachePreferL1);
		device_compute_cross_section_steptwo_block_hw<<<gridSize,blockSize>>>(g_freq, g_intens,g_nu,g_abs,hw,Npoints,N_ener,start_idx);
		//cudaDeviceSynchronize();
		//CheckCudaError("Step Two");
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
}

__host__ void execute_two_step_kernal_voigt_block(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif, int Npoints,int N_ener,int start_idx){
		
		int blockSize = 1024;
		int gridSize = (int)ceil((float)N_ener/blockSize);
		device_compute_cross_section_voigt_stepone<<<gridSize,blockSize>>>(g_energies,g_gns,g_nu,g_aif,N_ener);
		blockSize = VOIGT_BLOCK;
		gridSize = Npoints;//(int)ceil((float)Npoints/blockSize);
		//printf("gridSize = %d, blockSize = %d\n",gridSize,blockSize);
		 cudaFuncSetCacheConfig(device_compute_cross_section_steptwo_block, cudaFuncCachePreferL1);
		//device_compute_cross_section_voigt_steptwo_block<<<gridSize,blockSize>>>(g_freq, g_intens,g_nu,g_energies,Npoints,N_ener,start_idx);voigt profile
		//cudaDeviceSynchronize();
		//CheckCudaError("Step Two");
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
}
__host__ void execute_two_step_kernal_doppler_block(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif, int Npoints,int N_ener,int start_idx){
		
		int blockSize = 1024;
		int gridSize = (int)ceil((float)N_ener/blockSize);
		device_compute_cross_section_doppler_stepone<<<gridSize,blockSize>>>(g_energies,g_gns,g_nu,g_aif,N_ener);

		blockSize = BLOCK_SIZE;
		gridSize = Npoints;

		 cudaFuncSetCacheConfig(device_compute_cross_section_steptwo_block, cudaFuncCachePreferL1);
		device_compute_cross_section_doppler_steptwo_block<<<gridSize,blockSize>>>(g_freq, g_intens,g_nu,g_energies,Npoints,N_ener,start_idx);
		//cudaDeviceSynchronize();
		//CheckCudaError("Step Two");
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
}

__host__ void execute_two_step_kernal_doppler_block_temp(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double temp,double part, int Npoints,int N_ener,int start_idx,cudaStream_t stream){
		
		int blockSize = 1024;
		int gridSize = (int)ceil((float)N_ener/blockSize);
		device_compute_cross_section_doppler_stepone_temp<<<gridSize,blockSize,0,stream>>>(g_energies,g_gns,g_nu,g_aif,g_abs,temp,part,N_ener);

		blockSize = BLOCK_SIZE;
		gridSize = Npoints;

		 cudaFuncSetCacheConfig(device_compute_cross_section_doppler_steptwo_block_temp, cudaFuncCachePreferL1);
		device_compute_cross_section_doppler_steptwo_block_temp<<<gridSize,blockSize,0,stream>>>(g_freq, g_intens,g_nu,g_abs,temp,Npoints,N_ener,start_idx);
		//cudaDeviceSynchronize();
		//CheckCudaError("Step Two");
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
}

__host__ void execute_two_step_kernal_voigt_block(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_gamma,double* g_n ,int Npoints,int N_ener,int start_idx){
		
		int blockSize = 1024;
		int gridSize = (int)ceil((float)N_ener/blockSize);
		device_compute_cross_section_voigt_stepone<<<gridSize,blockSize>>>(g_energies,g_gns,g_nu,g_aif,g_gamma,g_n,N_ener);
		blockSize = VOIGT_BLOCK;
		gridSize = Npoints;//(int)ceil((float)Npoints/blockSize);
		//printf("gridSize = %d, blockSize = %d\n",gridSize,blockSize);
		 cudaFuncSetCacheConfig(device_compute_cross_section_voigt_steptwo_block, cudaFuncCachePreferL1);
		device_compute_cross_section_voigt_steptwo_block<<<gridSize,blockSize>>>(g_freq, g_intens,g_nu,g_energies,g_gamma,Npoints,N_ener,start_idx);
		//cudaDeviceSynchronize();
		//CheckCudaError("Step Two");
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
}

__host__ void execute_two_step_kernal_voigt_block_temp_press(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double* g_gamma,double* g_n ,double temp,double press,double part,int Npoints,int N_ener,int start_idx,cudaStream_t stream){
		
		int blockSize = 1024;
		int gridSize = (int)ceil((float)N_ener/blockSize);
		
		device_compute_cross_section_voigt_stepone_temp_press<<<gridSize,blockSize,0,stream>>>(g_energies,g_gns,g_nu,g_aif,g_abs,g_gamma,g_n,temp,press,part,N_ener);
		
		blockSize = VOIGT_BLOCK;
		gridSize = Npoints;

		 cudaFuncSetCacheConfig(device_compute_cross_section_voigt_steptwo_block_temp_press, cudaFuncCachePreferL1);

		device_compute_cross_section_voigt_steptwo_block_temp_press<<<gridSize,blockSize,0,stream>>>(g_freq, g_intens,g_nu,g_abs,g_gamma,temp,press,Npoints,N_ener,start_idx);
		//cudaDeviceSynchronize();
		//CheckCudaError("Step Two");
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
}


__host__ void execute_two_step_kernal_voigt(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif, int Npoints,int N_ener,int start_idx){
		printf("Using non-blocking method as Npoints is large %d > 100000\n",Npoints);
		int blockSize = 1024;
		int gridSize = (int)ceil((float)Npoints/blockSize);
		device_compute_cross_section_voigt_stepone<<<gridSize,blockSize>>>(g_energies,g_gns,g_nu,g_aif,N_ener);
		//cudaDeviceSynchronize();
		//CheckCudaError("Step One");
		blockSize = VOIGT_SHARED_SIZE;
		gridSize = (int)ceil((float)Npoints/blockSize);

		device_compute_cross_section_voigt_steptwo<<<gridSize,blockSize>>>(g_freq, g_intens,g_nu,g_energies,Npoints,N_ener,start_idx);
		//cudaDeviceSynchronize();
		//CheckCudaError("Step Two");

					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
}


__global__ void device_compute_cross_section(const double*  __restrict__ g_freq, double* g_cs,const double*  __restrict__ g_energies,const int*  __restrict__ g_gns,const int*  __restrict__ g_jF,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N,const int N_ener,const int start_idx){
	//The stored shared data
	__shared__ double l_energies[BLOCK_SIZE];
	__shared__ int l_gns[BLOCK_SIZE];
	__shared__ int l_jF[BLOCK_SIZE];
	__shared__ double l_nu[BLOCK_SIZE];
	__shared__ double l_aif[BLOCK_SIZE];
	
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int l_idx = threadIdx.x;
	int block_dim = blockDim.x;
	double cs_val = 0.0;
	double abscoef = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double temp_2,temp_3;
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N){
		freq = g_freq[start_idx+g_idx];
		cs_val = g_cs[start_idx+g_idx];
	}

	for(int i = 0; i < N_ener; i+=block_dim){
		l_energies[l_idx] = 0.0;
		l_gns[l_idx] = 0;
		l_nu[l_idx] = 1.0;
		l_aif[l_idx] = 0.0;
		l_jF[l_idx] = 0;
		if(i + l_idx < N_ener)
		{		
			//Store values in local memory
			l_energies[l_idx] = g_energies[i + l_idx];
			l_gns[l_idx] = g_gns[i + l_idx];
			l_nu[l_idx] = g_nu[i + l_idx];
			l_aif[l_idx] = g_aif[i + l_idx];
			l_jF[l_idx] = g_jF[i + l_idx];
		}
		__syncthreads();
		for(int j = 0; j < block_dim; j++){
			double nu_if = l_nu[j];
			
				abscoef = cross_constants.cmcoef*l_aif[j]*double(l_gns[j])*(2.0*double(l_jF[j]) + 1.0)
				*exp(-cross_constants.beta*l_energies[j])*(1.0-exp(-cross_constants.beta*nu_if))/
				(nu_if*nu_if*cross_constants.partition);
				temp_2=LN2PI/cross_constants.halfwidth;
				dfreq_ = l_nu[j]-freq;
				temp_3=exp(-cross_constants.ln2*(dfreq_/cross_constants.halfwidth)*(dfreq_/cross_constants.halfwidth));
				cs_val+=abscoef*temp_2*temp_3;
			
		}
		__syncthreads();
	}
	

	if(g_idx < N) g_cs[start_idx+g_idx]=cs_val;


}


__global__ void device_compute_cross_section_abscoef(const double*  __restrict__ g_freq, double* g_cs,const double*  __restrict__ g_energies,const int*  __restrict__ g_gns,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N,const int N_ener,const int start_idx){
	//The stored shared data
	__shared__ double l_nu[BLOCK_SIZE];
	__shared__ double l_abscoef[BLOCK_SIZE];
	
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int l_idx = threadIdx.x;
	int block_dim = BLOCK_SIZE;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double temp_2,temp_3;
	double nu_if;
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N){
		freq = g_freq[start_idx+g_idx];
		cs_val = g_cs[start_idx+g_idx];
	}

	//if(g_idx==9999)  printf("%12.6f\n",freq);	

	for(int i = 0; i < N_ener; i+=BLOCK_SIZE){
		l_nu[l_idx] = 1.0;
		l_abscoef[l_idx] = 0.0;
		if(i + l_idx < N_ener)
		{	
			double aif,gns,ei;
			//Store values in local memory
			ei = g_energies[i + l_idx];
			gns = double(g_gns[i + l_idx]);
			nu_if = g_nu[i + l_idx];
			
			aif = g_aif[i + l_idx];
			l_nu[l_idx] = nu_if;
			l_abscoef[l_idx] = cross_constants.cmcoef*aif*gns
				*exp(-cross_constants.beta*ei)*(1.0-exp(-cross_constants.beta*nu_if))/
				(nu_if*nu_if*cross_constants.partition);
		}
		__syncthreads();
		for(int j = 0; j < block_dim; j++){
			nu_if = l_nu[j];
						
			temp_2=LN2PI/cross_constants.halfwidth;
			dfreq_ = nu_if-freq;
			temp_3=exp(-LN2*(dfreq_/cross_constants.halfwidth)*(dfreq_/cross_constants.halfwidth));
			cs_val+=l_abscoef[j]*temp_2*temp_3;

			
		}
		__syncthreads();
	}
	

	if(g_idx < N) g_cs[start_idx+g_idx]=cs_val;


}

__global__ void device_compute_cross_section_warp_abscoef(const double*  __restrict__ g_freq, double* g_cs,const double*  __restrict__ g_energies,const int*  __restrict__ g_gns,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N,const int N_ener,const int start_idx){
	//The stored shared data
	__shared__ double l_nu[BLOCK_SIZE];
	__shared__ double l_abscoef[BLOCK_SIZE];
	
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int l_idx = threadIdx.x;
	int w_idx = l_idx % 32;
	
	int b_start = (threadIdx.x/32)*32;

	int block_dim = 32;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double temp_2,temp_3;
	double nu_if;
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N){
		freq = g_freq[start_idx+g_idx];
		cs_val = g_cs[start_idx+g_idx];
	}

	//if(g_idx==9999)  printf("%12.6f\n",freq);	

	for(int i = 0; i < N_ener; i+=block_dim){
		l_nu[l_idx] = 1.0;
		l_abscoef[l_idx] = 0.0;
		if(i + w_idx < N_ener)
		{	
			double aif,gns,ei;
			//Store values in local memory
			ei = g_energies[i + w_idx];
			gns = double(g_gns[i + w_idx]);
			nu_if = g_nu[i + w_idx];
			
			aif = g_aif[i + w_idx];
			l_nu[l_idx] = nu_if;
			l_abscoef[l_idx] = (LN2PI/cross_constants.halfwidth)*cross_constants.cmcoef*aif*gns
				*exp(-cross_constants.beta*ei)*(1.0-exp(-cross_constants.beta*nu_if))/
				(nu_if*nu_if*cross_constants.partition);
		}
		//__syncthreads();
		for(int j = b_start; j < (b_start+32); j++){
			nu_if = l_nu[j];
						
			//temp_2=cross_constants.ln2pi/cross_constants.halfwidth;
			dfreq_ = nu_if-freq;
			temp_3=exp(-cross_constants.ln2*(dfreq_/cross_constants.halfwidth)*(dfreq_/cross_constants.halfwidth));
			cs_val+=l_abscoef[j]*temp_3;

			
		}
		//__syncthreads();
	}
	

	if(g_idx < N) g_cs[start_idx+g_idx]=cs_val;


}

__global__ void device_compute_cross_section_stepone(double* g_energies,const int*  __restrict__ g_gns,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N_ener){
	//The stored shared data
	
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double ei,gns,nu_if,aif,abscoef;
	double temp_2 = 0.5/cross_constants.dfreq;//cross_constants.ln2pi/cross_constants.halfwidth;
	
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N_ener){
			//Store values in local memory
			ei = g_energies[g_idx];
			gns = g_gns[g_idx];
			nu_if = g_nu[g_idx];
			aif = g_aif[g_idx];
				
			abscoef= cross_constants.cmcoef*aif*gns
				*exp(-cross_constants.beta*ei)*(1.0-exp(-cross_constants.beta*nu_if))/
				(nu_if*nu_if*cross_constants.partition);
			if(nu_if==0)abscoef=0.0;
			g_energies[g_idx] = abscoef;
	}


}


__global__ void device_compute_cross_section_steptwo(const double*  __restrict__ g_freq, double* g_cs,const double*  __restrict__ g_nu,const double*  __restrict__ g_abscoef,const int N,const int N_ener,const int start_idx){
	//The stored shared data
	__shared__ double l_nu[SHARED_SIZE];
	__shared__ double l_abscoef[SHARED_SIZE];
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int l_idx = threadIdx.x;
	int block_dim = BLOCK_SIZE;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double hw = cross_constants.halfwidth*10.0;
	//double temp_2=cross_constants.ln2pi/cross_constants.halfwidth;
	//double temp_3 = -cross_constants.ln2*(1.0/(cross_constants.halfwidth*cross_constants.halfwidth));
	
	double x_ = SQRTLN2/cross_constants.halfwidth;
	double x0= x_*cross_constants.dfreq*0.5;
	//double nu_if;
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N){
		freq = g_freq[start_idx+g_idx];
		cs_val = g_cs[start_idx+g_idx];
	}

	//if(g_idx==9999)  printf("%12.6f\n",freq);	

	for(int i = 0; i < N_ener; i+=SHARED_SIZE){
		l_nu[l_idx] = 1.0;
		l_abscoef[l_idx] = 0.0;

		if(i + l_idx < N_ener)
		{	
			l_nu[l_idx] = g_nu[i + l_idx];
			l_abscoef[l_idx] = g_abscoef[i + l_idx];
		}
		
		l_nu[l_idx+ BLOCK_SIZE] = 1.0;
		l_abscoef[l_idx+ BLOCK_SIZE] = 0.0;
		if(i + l_idx + BLOCK_SIZE < N_ener)
		{	
			l_nu[l_idx+BLOCK_SIZE] = g_nu[i + l_idx+BLOCK_SIZE];
			l_abscoef[l_idx+BLOCK_SIZE] = g_abscoef[i + l_idx+BLOCK_SIZE];
		}
		__syncthreads();
		for(int j = 0; j < SHARED_SIZE; j++){
			//nu_if = ;
			double xp,xm,de;
			dfreq_ = l_nu[j]-freq;
			if(abs(dfreq_) > hw) continue;
			xp = x_*(dfreq_)+x0;
			xm = x_*(dfreq_)-x0;
			de = erf(xp)-erf(xm); 

			cs_val+=l_abscoef[j]*de;
			//*__expf(temp_3*dfreq_*dfreq_);

			
		}
		__syncthreads();
	}
	

	if(g_idx < N) g_cs[start_idx+g_idx]=cs_val;


}



__global__ void device_compute_cross_section_steptwo_block(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const int N,const int N_ener,const int start_idx){
	//The stored shared data
	//__shared__ double l_nu[BLOCK_SIZE];
	//__shared__ double l_abscoef[BLOCK_SIZE];
	__shared__ double l_cs_result[BLOCK_SIZE];
	//Get the global and local thread number
	int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double abscoef = 0.0;
	double hw = cross_constants.halfwidth;
	double dfreq = cross_constants.dfreq;
	double gammaG;
	double x0= SQRTLN2/hw*dfreq*0.5;

	freq = g_freq[start_idx + b_idx];
	//cs_val = g_cs[start_idx+g_idx];

	//if(g_idx==9999)  printf("%12.6f\n",freq);	
	l_cs_result[l_idx] = cs_val;
	for(int i = l_idx; i < N_ener; i+=BLOCK_SIZE){
		nu = 0.0;
		abscoef = 0.0;

		//Read value of nu
		nu = g_nu[i];

		dfreq_ = freq-nu;

		if(dfreq_ > hw*10.0) continue;
		if(dfreq_<-hw*10.0) break;

		gammaG =  SQRTLN2/hw;

		double xp,xm,de;
		
		xp = gammaG*(dfreq_)+x0;
		xm = gammaG*(dfreq_)-x0;

		

		de = erf(xp)-erf(xm); 


		//do work
		
				
		
	//	if(dfreq_<hw)continue;
	//	

		cs_val+=g_abscoef[i]*de*0.5/dfreq;					

	}
	//Store results into shared memory
	l_cs_result[l_idx] = cs_val;
	cs_val = 0;
	//Wait for everone to finish nicely
	__syncthreads();
	if(l_idx == 0){
		for(int i = 0; i < BLOCK_SIZE; i++)
			cs_val+=l_cs_result[i];
		
		g_cs[start_idx+b_idx]+=cs_val;		
	}

}

__global__ void device_compute_cross_section_steptwo_block_hw(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,double hw,const int N,const int N_ener,const int start_idx)
{
	//The stored shared data
	volatile __shared__ double l_nu[SHARED_SIZE];
	volatile __shared__ double l_abscoef[SHARED_SIZE];
	//__shared__ int leave[BLOCK_SIZE];
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	//__shared__ double l_cs_result[BLOCK_SIZE];
	//Get the global and local thread number
	//int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	int w_idx;
	int b_start = (threadIdx.x/32)*32;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double abscoef = 0.0;
       // int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double dfreq = cross_constants.dfreq;
	double gammaG =  SQRTLN2/hw;
	//int leave;
	double x0= SQRTLN2/hw*dfreq*0.5;

	//freq = g_freq[start_idx + b_idx];
	//cs_val = g_cs[start_idx+g_idx];

	//if(g_idx==9999)  printf("%12.6f\n",freq);	
	if(g_idx < N){
		freq = g_freq[start_idx+g_idx];
		//cs_val = g_cs[start_idx+g_idx];
	}

	//if(g_idx==9999)  printf("%12.6f\n",freq);	

	for(int i = 0; i < N_ener; i+=WARP_SIZE){
		l_nu[l_idx] =  1e100;
		l_abscoef[l_idx] = 0.0;
		w_idx = i + l_idx;

		if(i + l_idx < N_ener)
		{	
			l_nu[l_idx] = g_nu[w_idx];
			l_abscoef[l_idx] = g_abscoef[w_idx];
		}
		
		//l_nu[l_idx+ BLOCK_SIZE] = 1.0;
		//l_abscoef[l_idx+ BLOCK_SIZE] = 0.0;
		//if(i + l_idx + BLOCK_SIZE < N_ener)
			
		//	l_nu[l_idx+BLOCK_SIZE] = g_nu[i + l_idx+BLOCK_SIZE];
		//	l_abscoef[l_idx+BLOCK_SIZE] = g_abscoef[i + l_idx+BLOCK_SIZE];
		//}
		//__syncthreads();
		

		for(int j = 0; j < WARP_SIZE; j++){
			//nu_if = ;
		//Read value of nu
			nu = l_nu[b_start+j];

			dfreq_ = freq-nu;

			if(dfreq_ > hw*10.0) 
				continue;

			if(dfreq_<-hw*10.0){
				l_abscoef[l_idx] = 0.0;
				break;
			}			

			//gammaG

			double xp,xm,de;
		
			xp = gammaG*(dfreq_)+x0;
			xm = gammaG*(dfreq_)-x0;
			de = erf(xp)-erf(xm); 
			cs_val+=l_abscoef[b_start+j]*de*0.5/dfreq;
			//*__expf(temp_3*dfreq_*dfreq_);

			
		}
		//__syncthreads();
	}
	

	if(g_idx < N) g_cs[start_idx+g_idx]+=cs_val;

}

//////////////////////////////////////////////////////////
/*			Doppler				*/
//////////////////////////////////////////////////////////
__global__ void device_compute_cross_section_doppler_stepone(double* g_energies,const int*  __restrict__ g_gns,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N_ener){
	//The stored shared data
	
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double ei,gns,nu_if,aif,abscoef;
	
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N_ener){
			//Store values in local memory
			ei = g_energies[g_idx];
			gns = g_gns[g_idx];
			nu_if = g_nu[g_idx];
			aif = g_aif[g_idx];
				
			abscoef= CM_COEF*aif*gns
				*exp(-cross_constants.beta*ei)*(1.0-exp(-cross_constants.beta*nu_if))/
				(nu_if*nu_if*cross_constants.partition);
			if(nu_if==0)abscoef=0.0;
			g_energies[g_idx] = abscoef;
	}


}

__global__ void device_compute_cross_section_doppler_stepone_temp(const double* g_energies,const int*  g_gns,const double*  g_nu,const double*  g_aif,double* g_abscoef,const double temperature,const double partition, const int N_ener){
	//The stored shared data
	
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double ei,gns,nu_if,aif,abscoef;
	
	double beta = PLANCK*VELLGT/(BOLTZ*temperature);
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N_ener){
			//Store values in local memory
			ei = g_energies[g_idx];
			gns = g_gns[g_idx];
			nu_if = g_nu[g_idx];
			aif = g_aif[g_idx];
				
			abscoef= CM_COEF*aif*gns
				*exp(-beta*ei)*(1.0-exp(-beta*nu_if))/
				(nu_if*nu_if*partition);
			if(nu_if==0)abscoef=0.0;
			g_abscoef[g_idx] = abscoef;
	}


}


__global__ void device_compute_cross_section_doppler_steptwo_block(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const int N,const int N_ener,const int start_idx){
	//The stored shared data
	//__shared__ double l_nu[BLOCK_SIZE];
	//__shared__ double l_abscoef[BLOCK_SIZE];
	__shared__ double l_cs_result[BLOCK_SIZE];
	//Get the global and local thread number
	int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double abscoef = 0.0;
	double dpwcoeff = sqrt(2.0*BOLTZ*cross_constants.temperature*NA/((cross_constants.mean_mass)))/VELLGT;

	double gammaG;
	double x0= cross_constants.dfreq*0.5;

	freq = g_freq[start_idx + b_idx];
	//cs_val = g_cs[start_idx+g_idx];

	//if(g_idx==9999)  printf("%12.6f\n",freq);	
	l_cs_result[l_idx] = cs_val;
	for(int i = l_idx; i < N_ener; i+=BLOCK_SIZE){
		nu = 0.0;
		abscoef = 0.0;

		//Read value of nu
		nu = g_nu[i];

		dfreq_ = nu-freq;

		if(dfreq_ < -DOPPLER_CUTOFF) continue;
		if(dfreq_>DOPPLER_CUTOFF) break;

		gammaG =  SQRTLN2/(nu*dpwcoeff);

		double xp,xm,de;
		
		xp = gammaG*(abs(dfreq_)+x0);
		xm = gammaG*(abs(dfreq_)-x0);

		

		de = erf(xp)-erf(xm); 


		//do work
		
				
		
	//	if(dfreq_<hw)continue;
	//	

		cs_val+=g_abscoef[i]*de;					

	}
	//Store results into shared memory
	l_cs_result[l_idx] = cs_val;
	cs_val = 0;
	//Wait for everone to finish nicely
	__syncthreads();
	if(l_idx == 0){
		for(int i = 0; i < BLOCK_SIZE; i++)
			cs_val+=l_cs_result[i];
		
		g_cs[start_idx+b_idx]+=cs_val;		
	}

}

__global__ void device_compute_cross_section_doppler_steptwo_block_temp(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double temperature,const int N,const int N_ener,const int start_idx){
	//The stored shared data
	//__shared__ double l_nu[BLOCK_SIZE];
	//__shared__ double l_abscoef[BLOCK_SIZE];
	__shared__ double l_cs_result[BLOCK_SIZE];
	//Get the global and local thread number
	int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double abscoef = 0.0;
	double dpwcoeff = sqrt(2.0*BOLTZ*temperature*NA/((cross_constants.mean_mass)))/VELLGT;

	double gammaG;
	double x0= cross_constants.dfreq*0.5;

	freq = g_freq[start_idx + b_idx];
	//cs_val = g_cs[start_idx+g_idx];

	//if(g_idx==9999)  printf("%12.6f\n",freq);	
	l_cs_result[l_idx] = cs_val;
	for(int i = l_idx; i < N_ener; i+=BLOCK_SIZE){
		nu = 0.0;
		abscoef = 0.0;

		//Read value of nu
		nu = g_nu[i];

		dfreq_ = nu-freq;

		if(dfreq_ < -DOPPLER_CUTOFF) continue;
		if(dfreq_>DOPPLER_CUTOFF) break;

		gammaG =  SQRTLN2/(nu*dpwcoeff);

		double xp,xm,de;
		
		xp = gammaG*(dfreq_+x0);
		xm = gammaG*(dfreq_-x0);

		

		de = erf(xp)-erf(xm); 


		//do work
		
				
		
	//	if(dfreq_<hw)continue;
	//	

		cs_val+=g_abscoef[i]*de;					

	}
	//Store results into shared memory
	l_cs_result[l_idx] = cs_val;
	cs_val = 0;
	//Wait for everone to finish nicely
	__syncthreads();
	if(l_idx == 0){
		for(int i = 0; i < BLOCK_SIZE; i++)
			cs_val+=l_cs_result[i];
		
		g_cs[start_idx+b_idx]+=cs_val;		
	}

}


//////////////////////////////////////////////////////////
/*			VOIGT PROFILE			*/
//////////////////////////////////////////////////////////
__global__ void device_compute_cross_section_voigt_stepone(double* g_energies,const int*  __restrict__ g_gns,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N_ener){
	//The stored shared data
	
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double ei,gns,nu_if,aif,abscoef;
	double temp_2 = 1.0;//cross_constants.ln2pi/cross_constants.halfwidth;
	
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N_ener){
			//Store values in local memory
			ei = g_energies[g_idx];
			gns = g_gns[g_idx];
			nu_if = g_nu[g_idx];
			aif = g_aif[g_idx];
				
			abscoef= cross_constants.cmcoef*temp_2*aif*gns
				*exp(-cross_constants.beta*ei)*(1.0-exp(-cross_constants.beta*nu_if))/
				(nu_if*nu_if*cross_constants.partition);
			if(nu_if==0)abscoef=0.0;
			g_energies[g_idx] = abscoef;
			
	}


}

__global__ void device_compute_cross_section_voigt_stepone(double* g_energies,const int*  g_gns,const double*  g_nu,const double*  g_aif,double*  g_gamma,double*  g_n, const int N_ener){
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
			abscoef= cross_constants.cmcoef*aif*gns*ISQRTPI
				*exp(-cross_constants.beta*ei)*(1.0-exp(-cross_constants.beta*nu_if))/
				(nu_if*nu_if*cross_constants.partition);
			if(gns==-1) abscoef = aif;
			g_energies[g_idx] = abscoef;

			gammaL = g_gamma[g_idx]*pow(cross_constants.ref_temp/cross_constants.temperature,g_n[g_idx]);
			g_gamma[g_idx] = gammaL;

			//if(threadIdx.x == 0) printf("%14.2E   %14.2E\n",abscoef,gammaL) ;
			
	}


}

__global__ void device_compute_cross_section_voigt_stepone_temp_press(const double* g_energies,const int*  g_gns,const double*  g_nu,const double*  g_aif,double*  g_abscoef,double*  g_gamma,double*  g_n,double temperature, double pressure,double partition, const int N_ener){
	//The stored shared data
	
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double ei,gns,nu_if,aif,abscoef;
	double gammaL;
	double beta = PLANCK*VELLGT/(BOLTZ*temperature);
	//cross_constants.ln2pi/cross_constants.halfwidth;
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N_ener){
			//Store values in local memory

			nu_if = g_nu[g_idx];

			if(nu_if==0) nu_if = 1e-6;
			abscoef= CM_COEF*g_aif[g_idx]*g_gns[g_idx]*ISQRTPI
				*exp(-beta*g_energies[g_idx])*(1.0-exp(-beta*nu_if))/
				(nu_if*nu_if*partition);
			//if(gns==-1) abscoef = aif;
			g_abscoef[g_idx] = abscoef;
			
			gammaL = g_gamma[g_idx]*pow(cross_constants.ref_temp/temperature,g_n[g_idx]);
			g_gamma[g_idx] = gammaL;

			//if(threadIdx.x == 0) printf("%14.2E   %14.2E\n",abscoef,gammaL) ;
			
	}


}

__global__ void device_compute_cross_section_voigt_steptwo(const double* g_freq, double* g_cs,const double*  g_nu,const double*  g_abscoef,const int N,const int N_ener,const int start_idx){
	//The stored shared data
	__shared__ double l_nu[VOIGT_SHARED_SIZE];
	__shared__ double l_abscoef[VOIGT_SHARED_SIZE];
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	int l_idx = threadIdx.x;
	int block_dim = VOIGT_SHARED_SIZE;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double gammaG=0.05,gammaL=0.05,x,y;
	double dpwcoeff = sqrt(2.0*LN2*BOLTZ*cross_constants.temperature/(cross_constants.mean_mass))/VELLGT;
	//double nu_if;
	//if(g_idx == 0) printf("BLOCK_SIZE = %d\n",blockDim.x);
	if(g_idx < N){
		freq = g_freq[start_idx+g_idx];
		//cs_val = g_cs[start_idx+g_idx];
	}

	//if(g_idx==9999)  printf("%12.6f\n",freq);	

	for(int i = 0; i < N_ener; i+=VOIGT_SHARED_SIZE){
		l_nu[l_idx] = 1.0;
		l_abscoef[l_idx] = 0.0;

		if(i + l_idx < N_ener)
		{	
			l_nu[l_idx] = g_nu[i + l_idx];
			l_abscoef[l_idx] = g_abscoef[i + l_idx];
		}
		
		__syncthreads();
		for(int j = 0; j < VOIGT_SHARED_SIZE; j++){
			dfreq_=l_nu[j]-freq;
			gammaG = l_nu[j]*dpwcoeff;
			x =SQRTLN2*abs(dfreq_)/gammaG;
			y =SQRTLN2*gammaL/gammaG;
			double xxyy = x * x + y * y;
			
			//Algorithm 916
			if(xxyy < 100.0){
				cs_val+=l_abscoef[j]*SQRTLN2PI/(gammaG)*y*voigt_916(x,y,1.0);					
			}
else{
				//3-point gauss hermite
			cs_val+=l_abscoef[j]*(SQRTLN2PI/gammaG)*voigt_elevengausshermite(x,y,xxyy);
			}
			//*__expf(temp_3*dfreq_*dfreq_);

			
		}
		__syncthreads();
		


	}
	

	if(g_idx < N) g_cs[start_idx+g_idx]+=cs_val;


}


/*

__global__ void device_compute_cross_section_voigt_steptwo_block(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const int N,const int N_ener,const int start_idx){
	//The stored shared data
	//__shared__ double l_nu[BLOCK_SIZE];
	//__shared__ double l_abscoef[BLOCK_SIZE];
	__shared__ double l_cs_result[VOIGT_BLOCK];
	//Get the global and local thread number
	int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double gammaG=0.05,gammaL=0.05;
	double x,y;

	double dpwcoeff = sqrt(2.0*LN2*BOLTZ*cross_constants.temperature/(cross_constants.mean_mass))/VELLGT;

	//double temp_2=cross_constants.ln2pi/cross_constants.halfwidth;
	//double temp_3 = -cross_constants.ln2*(1.0/(cross_constants.halfwidth*cross_constants.halfwidth));

	freq = g_freq[start_idx + b_idx];
	//cs_val = g_cs[start_idx+g_idx];

	//if(g_idx==9999)  printf("%12.6f\n",freq);	
	l_cs_result[l_idx] = cs_val;
	for(int i = l_idx; i < N_ener; i+=VOIGT_BLOCK){
		nu = 0.0;
		//Read value of nu
		nu = g_nu[i];
		dfreq_ = nu-freq;
		if(dfreq_ < -500.0*gammaL)
			continue;
		if(dfreq_ > 500.0*gammaL)
			break;
		gammaG = nu*dpwcoeff;
		x =SQRTLN2*dfreq_/gammaG;
		y =SQRTLN2*gammaL/gammaG;
		double xxyy = x * x + y * y;

		
		////Algorithm 916
		if(xxyy < 100.0){
			cs_val+=g_abscoef[i]*SQRTLN2PI/(gammaG)*y*voigt_916(x,y,1.0);					
		}else{
			//3-point gauss hermite
			cs_val+=g_abscoef[i]*(SQRTLN2PI/gammaG)*voigt_threegausshermite(x,y,xxyy);
		}
			

	}
	//Store results into shared memory
	l_cs_result[l_idx] = cs_val;
	cs_val = 0;
	//Wait for everone to finish nicely
	__syncthreads();
	if(l_idx == 0){
		for(int i = 0; i < VOIGT_BLOCK; i++)
			cs_val+=l_cs_result[i];
		
		g_cs[start_idx+b_idx]+=cs_val;		
	}

}
*/

__global__ void device_compute_cross_section_voigt_steptwo_block_temp_press(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double*  g_gamma,const double temperature,const double pressure,const int N,const int N_ener,const int start_idx){
	//The stored shared data

	__shared__ double l_cs_result[VOIGT_BLOCK];
	//Get the global and local thread number
	int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double gammaG=0.05;
	double x,y;

	double dpwcoeff = sqrt(2.0*BOLTZ*temperature*NA/((cross_constants.mean_mass)))/VELLGT;
	double lorentz_cutoff = min(25.0*pressure,100.0);

	freq = g_freq[start_idx + b_idx];
	//cs_val = g_cs[start_idx+g_idx];
	//Lets find which energy we deal with


	//if(g_idx==9999)  printf("%12.6f\n",freq);	
	l_cs_result[l_idx] = cs_val;
	for(int i = l_idx; i < N_ener; i+=VOIGT_BLOCK){
		nu = 0.0;
		//Read value of nu
		nu = g_nu[i];
		dfreq_ = nu-freq;
		
		if(dfreq_ < -lorentz_cutoff)
			continue;
		if(dfreq_ > lorentz_cutoff)
			break; //We are done here let another queued block do something
		gammaG = 1.0/(nu*dpwcoeff);
		x =abs(dfreq_)*gammaG;
		y =g_gamma[i]*gammaG;

		double xxyy = x * x + y * y;
		//double voigt;// = voigt_916(x,y,0.9);
		//voigt = humlick(x,y);
		/*
		
		////Algorithm 916///
		if(xxyy < 100.0){
			voigt = voigt_916(x,y,0.9);
			//cs_val+=g_abscoef[i]*voigt_check*gammaG*ISQRTPI;					
		}else if(xxyy < 1.0e6){
			//3-point gauss hermite
			voigt = 1.1181635900*y/(PI*xxyy) + 2.0*0.2954089751*(xxyy + 1.5)/( (xxyy+1.5)*(xxyy+1.5) - 4*x*x*1.5);
			//cs_val+=g_abscoef[i]*ISQRTPI*gammaG;
		}else{
			voigt = y/(PI*xxyy); //This is basically lorentz
			//cs_val+= g_abscoef[i]*ISQRTPI*gammaG;
		}
		*/

		cs_val+=g_abscoef[i]*voigt_threegausshermite(x,y,xxyy)*gammaG;//humlick(x,y)*gammaG;
		

	}
	//Store results into shared memory
	l_cs_result[l_idx] = cs_val;
	cs_val = 0;
	//Wait for everone to finish nicely
	__syncthreads();
	if(l_idx == 0){
		for(int i = 0; i < VOIGT_BLOCK; i++)
			cs_val+=l_cs_result[i];
		
		g_cs[start_idx+b_idx]+=cs_val;		
	}

}


__global__ void device_compute_cross_section_voigt_steptwo_block(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double*  g_gamma,const int N,const int N_ener,const int start_idx){
	//The stored shared data

	__shared__ double l_cs_result[VOIGT_BLOCK];
	//Get the global and local thread number
	int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double gammaG=0.05;
	double x,y;

	double dpwcoeff = sqrt(2.0*BOLTZ*cross_constants.temperature*NA/((cross_constants.mean_mass)))/VELLGT;
	double lorentz_cutoff = min(25.0*cross_constants.pressure,100.0);

	freq = g_freq[start_idx + b_idx];
	//cs_val = g_cs[start_idx+g_idx];
	//Lets find which energy we deal with


	//if(g_idx==9999)  printf("%12.6f\n",freq);	
	l_cs_result[l_idx] = cs_val;
	for(int i = l_idx; i < N_ener; i+=VOIGT_BLOCK){
		nu = 0.0;
		//Read value of nu
		nu = g_nu[i];
		dfreq_ = nu-freq;
		
		if(dfreq_ < -lorentz_cutoff)
			continue;
		if(dfreq_ > lorentz_cutoff)
			break; //We are done here let another queued block do something
		gammaG = 1.0/(nu*dpwcoeff);
		x =abs(dfreq_)*gammaG;
		y =g_gamma[i]*gammaG;

		//double xxyy = x * x + y * y;
		//double voigt;// = voigt_916(x,y,0.9);
		//voigt = humlick(x,y);
		/*
		
		////Algorithm 916///
		if(xxyy < 100.0){
			voigt = voigt_916(x,y,0.9);
			//cs_val+=g_abscoef[i]*voigt_check*gammaG*ISQRTPI;					
		}else if(xxyy < 1.0e6){
			//3-point gauss hermite
			voigt = 1.1181635900*y/(PI*xxyy) + 2.0*0.2954089751*(xxyy + 1.5)/( (xxyy+1.5)*(xxyy+1.5) - 4*x*x*1.5);
			//cs_val+=g_abscoef[i]*ISQRTPI*gammaG;
		}else{
			voigt = y/(PI*xxyy); //This is basically lorentz
			//cs_val+= g_abscoef[i]*ISQRTPI*gammaG;
		}
		*/

		cs_val+=g_abscoef[i]*humlick(x,y)*gammaG;
		

	}
	//Store results into shared memory
	l_cs_result[l_idx] = cs_val;
	cs_val = 0;
	//Wait for everone to finish nicely
	__syncthreads();
	if(l_idx == 0){
		for(int i = 0; i < VOIGT_BLOCK; i++)
			cs_val+=l_cs_result[i];
		
		g_cs[start_idx+b_idx]+=cs_val;		
	}

}


__global__ void device_compute_cross_section_voigt_single(double* g_freq,double* g_cs,const double   nu_if,const double ei, const double gns,const double  aif,const double  g_gamma, const double pressure,const double p_n, const double temperature, const double partition, const int N){
	//The stored shared data

	
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double gammaG = sqrt(2.0*BOLTZ*temperature*NA/((cross_constants.mean_mass)))/VELLGT;
	double gammaL = g_gamma*pow(cross_constants.ref_temp/temperature,p_n);
	gammaG = 1.0/(nu_if*gammaG);
	double x,y,dfreq_;
	double abscoef;
	double beta = BETA_PREF/temperature;
	
	for(int g_idx = blockIdx.x * blockDim.x + threadIdx.x; g_idx < N; g_idx += blockDim.x * gridDim.x){
		dfreq_ = nu_if-g_freq[g_idx];
		abscoef= cross_constants.cmcoef*ISQRTPI*aif*gns*exp(-beta*ei)*(1.0-exp(-beta*nu_if))/(nu_if*nu_if*partition);
		x =abs(dfreq_)*gammaG;
		y =gammaL*gammaG;
		g_cs[g_idx]+=abscoef*humlick(x,y)*gammaG;
	}

}

