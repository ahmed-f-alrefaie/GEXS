#include "cross_structs.cuh"
#include "cross_kernal_v2.cuh"
#include "cuda_utils.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include<stdio.h>
#include<stdlib.h>
#include <cmath>
#include <cuComplex.h>
#include <thrust/complex.h>
#include <cub/cub.cuh>
#include "defines.h"

__constant__ cross_section_data cross_constants;	
//Computing temperature and pressure


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



__inline__ __device__
int warpAllReduceSum(int val) {
  for (int mask = warpSize/2; mask > 0; mask /= 2) 
    val += __shfl_xor(val, mask);
  return val;
}


__host__ void copy_intensity_info(cross_section_data* cross_inf)
{
	//void* ptr;
	//cudaGetSymbolAddress ( &ptr, int_info );

	cudaMemcpyToSymbol(cross_constants, (void*)cross_inf, sizeof(cross_section_data),0,cudaMemcpyHostToDevice);
};

//--------------------------------------Compute spectroscopic quantities-----------------------------------------------
__global__ void device_compute_abscoefs(const double* g_energies,const int*  g_gns,const double*  g_nu,const double*  g_aif,double* g_abscoef,const double temperature,const double partition, const int N_ener){
	//The stored shared data
	
	//Get the global and local thread number
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double ei,gns,nu_if,aif,abscoef;
	
	double beta = -PLANCK*VELLGT/(BOLTZ*temperature);
	//if(g_idx == 0) printf("partition = %12.6f\n",cross_constants.partition);
	if(g_idx < N_ener){
			//Store values in local memory
			ei = g_energies[g_idx];
			gns = g_gns[g_idx];
			nu_if = g_nu[g_idx];
			aif = g_aif[g_idx];
				
			abscoef= CM_COEF*aif*gns
				*exp(beta*ei)*(1.0-exp(beta*nu_if))/
				(nu_if*nu_if*partition);
			g_abscoef[g_idx] = abscoef;
	}


}

__global__ void  device_compute_pressure(double*  g_gamma,const double*  g_n,const double temperature,const double pressure, const int N_ener){

	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	double gammaL;
	if(g_idx < N_ener){
			
			gammaL = g_gamma[g_idx]*pow(cross_constants.ref_temp/temperature,g_n[g_idx])*(pressure/cross_constants.ref_press);
			g_gamma[g_idx] = gammaL;
			
	}
}



__global__ void device_compute_lorentzian(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,double hw,const int N,const int N_ener,const int start_idx)
{
	//The stored shared data
	volatile __shared__ double l_nu[SHARED_SIZE];
	volatile __shared__ double l_abscoef[SHARED_SIZE];
	//volatile __shared__ int l_leave[SHARED_SIZE];
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
	double de = 0.0;
	int leave=0;

	if(g_idx < N){
		freq = g_freq[start_idx+g_idx];
	}


	for(int i = 0; i < N_ener; i+=WARP_SIZE){
		l_nu[l_idx] =  1e100;
		l_abscoef[l_idx] = 0.0;
		//l_leave[l_idx] = 0;
		w_idx = i + l_idx;

		if(i + l_idx < N_ener)
		{	
			l_nu[l_idx] = g_nu[w_idx];
			dfreq_ = freq-nu;
			if(dfreq_ > 10.0*hw){
				leave = 1;
			}
			//Do we have any transitions within the warp range?
			if(warpAllReduceSum(leave)==WARP_SIZE)
				continue;
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

			//if(dfreq_ > hw*10.0) 
			//	continue;

			if(dfreq_<-hw*10.0){
				//l_leave[l_idx] = 1;
				leave=1;
				
			}			

			//gammaG
			de =dfreq_*dfreq_ + hw*hw;
			
			cs_val+=l_abscoef[b_start+j]*IPI*hw/de;
			//*__expf(temp_3*dfreq_*dfreq_);

			
		}
		if(warpAllReduceSum(leave)==WARP_SIZE)
			break;
		//__syncthreads();
	}
	

	if(g_idx < N) g_cs[start_idx+g_idx]+=cs_val;

}

__global__ void device_compute_doppler(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double temperature,const int N,const int N_ener,const int start_idx)
{

	typedef cub::WarpReduce<int> WarpReduceI;
	//The stored shared data
	volatile __shared__ double l_nu[SHARED_SIZE];
	volatile __shared__ double l_abscoef[SHARED_SIZE];
	//volatile __shared__ double l_freq[SHARED_SIZE];
	//volatile __shared__ double l_result[SHARED_SIZE];
	__shared__ typename WarpReduceI::TempStorage leave_warp[4];
	//__shared__ int leave[BLOCK_SIZE];
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	//__shared__ double l_cs_result[BLOCK_SIZE];
	//Get the global and local thread number
	//int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	int w_idx;
	int warp_id = threadIdx.x/32;
	int b_start = (threadIdx.x/32)*32;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double gammaG;
	double x0= cross_constants.dfreq*0.5;
	double dpwcoeff = sqrt(2.0*BOLTZ*temperature*NA/((cross_constants.mean_mass)))/VELLGT;
	int leave = 0;
	if(g_idx < N){
		//freq = g_freq[start_idx+g_idx];
		freq = g_freq[start_idx+g_idx]; 
		//cs_val = g_cs[start_idx+g_idx];
	}

	//if(g_idx==9999)  printf("%12.6f\n",freq);	

	for(int i = 0; i < N_ener; i+=WARP_SIZE){
		l_nu[l_idx] =  1e100;
		l_abscoef[l_idx] = 0.0;
		leave=0;
		w_idx = i + l_idx;

		if(i + l_idx < N_ener)
		{	
			l_nu[l_idx] = g_nu[w_idx];
			l_abscoef[l_idx] = g_abscoef[w_idx];
			dfreq_ = freq-l_nu[l_idx];
			if(dfreq_ > DOPPLER_CUTOFF){
				leave = 1;
			}
		}else{
			leave = 1;
		}
		if(WarpReduceI(leave_warp[warp_id]).Sum(leave)>=31)
			continue;
		leave = 0;
		for(int j = 0; j < WARP_SIZE; j++){

			nu = l_nu[b_start+j];

			dfreq_ = freq-nu;

			//if(dfreq_ > DOPPLER_CUTOFF) 
			//	continue;
			//Do we have any transitions left within the warp range?
			
			
			gammaG= SQRTLN2/(nu*dpwcoeff);
			double xp,xm,de;
		
			xp = gammaG*(dfreq_+x0);
			xm = gammaG*(dfreq_-x0);
			de = erf(xp)-erf(xm); 
			cs_val+=l_abscoef[b_start+j]*de;
			//*__expf(temp_3*dfreq_*dfreq_);

			
		}
		if(dfreq_<-DOPPLER_CUTOFF){
				//l_abscoef[l_idx] = 0.0;
				leave=1;
				//break;
		}
		if(WarpReduceI(leave_warp[warp_id]).Sum(leave)>=31)
			break;
	}
	

	if(g_idx < N) g_cs[start_idx+g_idx]+=cs_val;

}

__global__ void device_compute_doppler_block(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double temperature,const int N,const int N_ener,const int start_idx)
{
	typedef cub::BlockReduce<double, DOPPLER_SIZE> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	//__shared__ double l_cs_result[BLOCK_SIZE];
	//Get the global and local thread number
	int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	double cs_val = 0.0;
	double final_cs = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double abscoef = 0.0;
	double dpwcoeff = (temperature*cross_constants.dpwcoeff);

	double gammaG;
	double x0;

	freq = g_freq[start_idx + b_idx];
	//cs_val = g_cs[start_idx+g_idx];

	//if(g_idx==9999)  printf("%12.6f\n",freq);	
	//l_cs_result[l_idx] =  0.0;

	for(int i = l_idx; i < N_ener; i+=DOPPLER_SIZE){
		nu = 0.0;
		abscoef = 0.0;

		//Read value of nu
		nu = g_nu[i];

		dfreq_ = freq-nu;

		if(dfreq_ > DOPPLER_CUTOFF) continue;
		if(dfreq_<-DOPPLER_CUTOFF) break;

		gammaG =  SQRTLN2/(dpwcoeff*nu);
		x0 = gammaG*cross_constants.dfreq*0.5;
		double xp,xm,de;
		
		xp = gammaG*(dfreq_)+x0;
		xm = gammaG*(dfreq_)-x0;

		

		de = erf(xp)-erf(xm); 


		//do work
		
				
		
	//	if(dfreq_<hw)continue;
	//	

		cs_val+=g_abscoef[i]*de;					

	}
	//Store results into shared memory
	//l_cs_result[l_idx] = cs_val;
	//cs_val = 0;
	//Wait for everone to finish nicely
	__syncthreads();
	final_cs = BlockReduce(temp_storage).Sum(cs_val);
	if(l_idx == 0){
		//for(int i = 0; i < BLOCK_SIZE; i++)
		//	cs_val+=l_cs_result[i];
		
		g_cs[start_idx+b_idx]+=final_cs*0.5/cross_constants.dfreq;		
	}

}
__global__ void device_compute_voigt(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double*  g_gamma,const double temperature,const double pressure,const double lorentz_cutoff,const int N,const int N_ener,const int start_idx){
	//The stored shared data
	//typedef cub::BlockReduce<double, VOIGT_BLOCK> BlockReduce;
	//__shared__ typename BlockReduce::TempStorage temp_storage;
	__shared__ double l_cs_result[VOIGT_BLOCK];
	//__shared__ double l_correction[VOIGT_BLOCK];
	//Get the global and local thread number
	int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double gammaG=0.05;
	double x,y;
	double abscoef;
	double aggregate = 0.0;
	double dpwcoeff = temperature*cross_constants.dpwcoeff;
	
	freq = g_freq[start_idx + b_idx];

	//Lets find which energy we deal with


	//if(g_idx==9999)  printf("%12.6f\n",freq);	
	l_cs_result[l_idx] = 0.0;
	//l_correction[l_idx] = 0.0;
	for(int i = l_idx; i < N_ener; i+=VOIGT_BLOCK){
		nu = 0.0;
		//Read value of nu
		nu = g_nu[i];
		dfreq_ = nu-freq;
		if(dfreq_ < -lorentz_cutoff)
			continue;
		if(dfreq_ > lorentz_cutoff)
			break; //We are done here let another queued block do something

		//abscoef = g_abscoef[i];	
		gammaG = SQRTLN2/(nu*dpwcoeff);
		x = dfreq_*gammaG;
	
		//if(cs_val > 0){
			//Lets check contribution, if its close to epsilon then dont bother
		//	if(abscoef/cs_val < 1e-12)
		//		continue;
		//}		
				


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
		cs_val+= g_abscoef[i]*humlick(x,y)*gammaG;
		//x = cs_val + y;
		//correction = (x - cs_val) - y; 
		//cs_val=x;
		

	}
	
	//Store results into shared memory
	l_cs_result[l_idx] = cs_val;
	cs_val = 0;
	//Wait for everone to finish nicely
	__syncthreads();
	if(l_idx == 0){
		for(int i = 0; i < VOIGT_BLOCK; i++)
			cs_val+=l_cs_result[i];
		
		g_cs[start_idx+b_idx]+=cs_val*ISQRTPI;		
	}
//	aggregate = BlockReduce(temp_storage).Sum(cs_val);	
	//if(l_idx==0)g_cs[start_idx+b_idx]+=aggregate*ISQRTPI; //cs_val;		
	

}

__global__ void device_compute_voigt_exact(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double*  g_gamma,const double temperature,const double pressure,const double lorentz_cutoff,const int N,const int N_ener,const int start_idx){
	//The stored shared data
	typedef cub::BlockReduce<double, VOIGT_BLOCK> BlockReduce;
	__shared__ typename BlockReduce::TempStorage temp_storage;
	//__shared__ double l_cs_result[VOIGT_BLOCK];
	//__shared__ double l_correction[VOIGT_BLOCK];
	//Get the global and local thread number
	int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	volatile double cs_val = 0.0;
	volatile double dfreq_=0.0;
	volatile double freq = 0.0;
	volatile double nu = 0.0;
	volatile double gammaD=0.05;
	volatile double gammaL=0.01;
	volatile double x,y,z,fp,voigt;
	volatile double abscoef;
	volatile double aggregate = 0.0;
	volatile double dpwcoeff = temperature*cross_constants.dpwcoeff;
	
	freq = g_freq[start_idx + b_idx];

	//Lets find which energy we deal with


	//if(g_idx==9999)  printf("%12.6f\n",freq);	
	//l_cs_result[l_idx] = 0.0;

	//l_correction[l_idx] = 0.0;
	for(int i = l_idx; i < N_ener; i+=VOIGT_BLOCK){
		nu = 0.0;
		//Read value of nu
		nu = g_nu[i];
		dfreq_ = nu-freq;
		if(dfreq_ < -lorentz_cutoff)
			continue;
		if(dfreq_ > lorentz_cutoff)
			break; //We are done here let another queued block do something

		//abscoef = g_abscoef[i];	
		gammaD = SQRTLN2/(nu*dpwcoeff);
		//gammaL = g_gamma[i];
		
		//volatile double v1 = exp(4.0*PI*gammaL*gammaL/(gammaD*gammaD));
		//volatile double v2 = exp(pdfreq*pdfreq/(gammaD*gammaD));
		//volatile double v3 = cos((4.0*PI*gammaL*gammaL*SQRTLN2*pdfreq*pdfreq)/(gammaD*gammaD));

		x =dfreq_*gammaD;
		y =g_gamma[i]*gammaD;
		z = (nu + freq)*gammaD;
		fp =  SQRTPI*gammaD;
		volatile double ex2 = exp(-x * x);

		if(x==0){
			voigt = erfcx(y);
		}else if (y==0){
			voigt = ex2;
		}else{
			
			volatile double ez2 = exp(-z * z);
			volatile double ey2 = exp(4.0*PI*y * y);

			volatile double v1 = ey2*cos(y*z);
			volatile double v2 = ey2*cos(x*z);

			volatile double v3 = v1/ex2 + v2/ez2;
			voigt = fp*v3;


		}
			


		/*volatile double yy = y*y;
		volatile double xx = x*x;
		volatile double zz = z*z;
		
		volatile double v1 = exp(4.0*PI*yy - zz);
		volatile double v2 = cos(4.0*PI*y*z);
		volatile double v3 = exp(4.0*PI*yy - xx);
		volatile double v4 = exp(4.0*PI*yy - xx);
		
		voigt = v1*v2 + v3*v4;
		if(voigt != voigt){
			voigt = 0.0;
		}
		*/

			//Exact form
		

		cs_val+= g_abscoef[i]*voigt;
		//x = cs_val + y;
		//correction = (x - cs_val) - y; 
		//cs_val=x;
		

	}
	
	//Store results into shared memory
	//l_cs_result[l_idx] = cs_val;
	//l_correction[l_idx] = correction;
	//cs_val
	//Wait for everone to finish nicely
	__syncthreads();
	//if(l_idx == 0){
		//cs_val = g_cs[start_idx+b_idx];
		//correction = 0;		
		//for(int i = 0; i < VOIGT_BLOCK; i++)
		//	correction += l_correction[i];
		/*for(int i = 0; i < VOIGT_BLOCK; i++){
			y = l_cs_result[i] - correction;
			x = cs_val + y;
			correction = (x - cs_val) - y;
			cs_val=x;
		}*/
	//}
	aggregate = BlockReduce(temp_storage).Sum(cs_val);	
	if(l_idx==0)g_cs[start_idx+b_idx]+=aggregate*cross_constants.dfreq; //cs_val;		
	

}


__global__ void device_compute_voigt_II(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double*  g_gamma,const double temperature,const double pressure,const double lorentz_cutoff,const int N,const int N_ener,const int start_idx){
	//The stored shared data
	//typedef cub::WarpReduce<int> WarpReduceI;
	__shared__ double l_nu[VOIGT_BLOCK];
	__shared__ double l_abscoef[VOIGT_BLOCK];
	__shared__ double l_gamma[VOIGT_BLOCK];
	//__shared__ int l_leave[VOIGT_BLOCK];
	//__shared__ int l_continue[VOIGT_BLOCK];
	//typedef cub::BlockReduce<int, VOIGT_BLOCK> BlockReduce;
	//__shared__ typename BlockReduce::TempStorage temp_storage;
	//volatile __shared__ double l_abcissa[SHARED_SIZE];
	//volatile __shared__ double l_abcissa[SHARED_SIZE];
	//__shared__ int leave[BLOCK_SIZE];
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	//__shared__ double l_cs_result[BLOCK_SIZE];
	//Get the global and local thread number
	//int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double gammaG;
	double dpwcoeff = temperature*cross_constants.dpwcoeff;
//	int leave = 0;
//	int continue_cal = 0;
	double x,y;

	if(g_idx < N){
		freq = g_freq[start_idx+g_idx];
		//cs_val = g_cs[start_idx+g_idx];
	}

	//if(g_idx==9999)  printf("%12.6f\n",freq);	

	for(int i = 0; i < N_ener; i+=VOIGT_BLOCK){
		l_nu[l_idx] =  1e100;
		l_abscoef[l_idx] = 0.0;
		//leave=1;
		int w_idx = i + l_idx;
		//l_leave[l_idx] = 0;
		//
		if(w_idx < N_ener)
		{	
			l_nu[l_idx] = g_nu[w_idx];
			//l_leave[l_idx] = 1;
			//l_continue[l_idx] = 1;
			//dfreq_ = freq-nu;
		/*	if(dfreq_ < -LORENTZ_CUTOFF){
				l_leave[l_idx] = 0;
			}else if (dfreq_ > LORENTZ_CUTOFF){
				l_continue[l_idx] = 0;
			}else{
			
			//Do we have any transitions within the warp range?
			//if(warpAllReduceSum(leave)==WARP_SIZE)
			//	continue;

			}*/
			l_abscoef[l_idx] = g_abscoef[w_idx];
			l_gamma[l_idx] = g_gamma[w_idx];
		}
		
		//if(BlockReduce(temp_storage).Sum(leave)==0)
		//	break;
		__syncthreads();
		/*leave = 0;
		continue_cal = 0;
		
		for(int j = 0; j < VOIGT_BLOCK; j++){
			continue_cal += l_continue[j];
			leave+=l_leave[j];
		}

		if(leave == 0)
			break;
		if(continue_cal==0)
			continue;
		*/
		for(int j = 0; j < VOIGT_BLOCK; j++){

			nu = l_nu[j];

			dfreq_ = freq-nu;

			if(dfreq_ < -LORENTZ_CUTOFF) 
				break;
			if(dfreq_ > LORENTZ_CUTOFF)
				continue;
			
			//Do we have any transitions left within the warp range?
			//if(dfreq_>-lorentz_cutoff){
			//	l_abscoef[l_idx] = 0.0;
			//	leave=0;
			//	//break;
			//}		
			gammaG= SQRTLN2/(nu*dpwcoeff);	
			x =abs(dfreq_)*gammaG;
			y =l_gamma[j]*gammaG;
			
			cs_val+=l_abscoef[j]*humlick(x,y)*gammaG*ISQRTPI;
			//*__expf(temp_3*dfreq_*dfreq_);

			
		}
		//if(WarpReduceI(leave_warp[warp_id]).Sum(leave)==31)
		//	break;
		__syncthreads();
	}
	

	if(g_idx < N) g_cs[start_idx+g_idx]+=cs_val;



}

__global__ void device_compute_voigt_quad(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double*  g_gamma,const double temperature,const double pressure,const double lorentz_cutoff,const int N,const int N_ener,const int start_idx){
	//The stored shared data

	typedef cub::WarpReduce<int> WarpReduceI;
	typedef cub::WarpReduce<double> WarpReduceD;

	volatile __shared__ double l_nu[SHARED_SIZE];
	volatile __shared__ double l_abscoef[SHARED_SIZE];
	volatile __shared__ double l_gamma[SHARED_SIZE];
	
	__shared__ typename WarpReduceI::TempStorage leave_warp[4];
	volatile __shared__ double l_abcissa[SHARED_SIZE];
	volatile __shared__ double l_weight[SHARED_SIZE];
	__shared__ typename WarpReduceD::TempStorage cs_warp[4];
	//__shared__ int leave[BLOCK_SIZE];
	int g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	//__shared__ double l_cs_result[BLOCK_SIZE];
	//Get the global and local thread number
	//int b_idx = blockIdx.x;
	int l_idx = threadIdx.x;
	int w_idx;
	int b_start = (threadIdx.x/32)*32;
	int warp_id = threadIdx.x/32;
	double cs_val = 0.0;
	double dfreq_=0.0;
	double freq = 0.0;
	double nu = 0.0;
	double gammaG;
	double warp_cs = 0.0;
	double dpwcoeff = sqrt(2.0*BOLTZ*temperature*NA/((cross_constants.mean_mass)))/VELLGT;
	int leave = 0;
	double x,y;


	if(g_idx < N){
		freq = g_freq[start_idx+g_idx];
		//cs_val = g_cs[start_idx+g_idx];
	}

	//if(g_idx==9999)  printf("%12.6f\n",freq);	

	for(int i = 0; i < N_ener; i+=WARP_SIZE){
		l_nu[l_idx] =  1e100;
		l_abscoef[l_idx] = 0.0;
		leave=0;
		w_idx = i + l_idx;
		
		if(i + l_idx < N_ener)
		{	
			l_nu[l_idx] = g_nu[w_idx];
			
			dfreq_ = freq-nu;
			if(dfreq_ > DOPPLER_CUTOFF){
				leave = 1;
			}
			//Do we have any transitions within the warp range?
			if(WarpReduceI(leave_warp[warp_id]).Sum(leave)==WARP_SIZE)
				continue;
			l_abscoef[l_idx] = g_abscoef[w_idx];
			l_gamma[l_idx] = g_gamma[w_idx];
		}
		leave = 0;
		for(int j = 0; j < WARP_SIZE; j++){
			warp_cs = 0.0;
			nu = l_nu[b_start+j];

			dfreq_ = freq-nu;

			//if(dfreq_ > DOPPLER_CUTOFF) 
			//	continue;
			//Do we have any transitions left within the warp range?
			if(dfreq_<-lorentz_cutoff){
				//l_abscoef[l_idx] = 0.0;
				leave=1;
				//break;
			}		
			gammaG= SQRTLN2/(nu*dpwcoeff);	
			x =abs(dfreq_)*gammaG;
			y =l_gamma[b_start+j]*gammaG;
			


			
			cs_val+=l_abscoef[b_start+j]*humlick(x,y)*gammaG*ISQRTPI;
			//*__expf(temp_3*dfreq_*dfreq_);

			
		}
		if(WarpReduceI(leave_warp[warp_id]).Sum(leave)>=WARP_SIZE)
			break;
		//__syncthreads();
	}
	

	if(g_idx < N) g_cs[start_idx+g_idx]+=cs_val;



}


//Lorentzian
__host__ void gpu_compute_lorentzian_profile(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double temp,double part,double hw, int Npoints,int N_ener,int start_idx,cudaStream_t stream){
		int blockSize = 1024;
		int gridSize = (int)ceil((float)N_ener/blockSize);
		device_compute_abscoefs<<<gridSize,blockSize,0,stream>>>(g_energies,g_gns,g_nu,g_aif,g_abs,temp,part,N_ener);

		blockSize = SHARED_SIZE;
		gridSize = (int)ceil((float)Npoints/blockSize);

		device_compute_lorentzian<<<gridSize,blockSize,0,stream>>>(g_freq, g_intens,g_nu,g_abs,hw,Npoints,N_ener,start_idx);


}
//Gaussian
__host__ void gpu_compute_gaussian_profile(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double temp,double part,double hw, int Npoints,int N_ener,int start_idx,cudaStream_t stream){
}
//Doppler
__host__ void gpu_compute_doppler_profile(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double temp,double part, int Npoints,int N_ener,int start_idx,cudaStream_t stream){
		int blockSize = 1024;
		int gridSize = (int)ceil((float)N_ener/blockSize);
		device_compute_abscoefs<<<gridSize,blockSize,0,stream>>>(g_energies,g_gns,g_nu,g_aif,g_abs,temp,part,N_ener);

		blockSize = DOPPLER_SIZE;
		gridSize = Npoints;//(int)ceil((float)Npoints/blockSize);
cudaFuncSetCacheConfig(device_compute_doppler_block, cudaFuncCachePreferL1);
		device_compute_doppler_block<<<gridSize,blockSize,0,stream>>>(g_freq, g_intens,g_nu,g_abs,sqrt(temp),Npoints,N_ener,start_idx);
}
//Voigt
__host__ void gpu_compute_voigt_profile(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double* g_gamma,double* g_n ,double temp,double press,double part,int Npoints,int N_ener,int start_idx,cudaStream_t stream){

		int blockSize = 1024;
		int gridSize = (int)ceil((float)N_ener/(float)blockSize);
		device_compute_abscoefs<<<gridSize,blockSize,0,stream>>>(g_energies,g_gns,g_nu,g_aif,g_abs,temp,part,N_ener);
		device_compute_pressure<<<gridSize,blockSize,0,stream>>>(g_gamma,g_n ,temp,press,N_ener);
		//device_compute_abscoefs<<<gridSize,blockSize,0,stream>>>(g_energies,g_gns,g_nu,g_aif,g_abs,temp,part,N_ener);
				
		//blockSize = VOIGT_BLOCK;
		//gridSize = (int)ceil((float)Npoints/(float)blockSize);
		//device_compute_voigt_II<<<gridSize,blockSize,0,stream>>>(g_freq, g_intens,g_nu,g_abs,g_gamma,sqrt(temp),press,LORENTZ_CUTOFF,Npoints,N_ener,start_idx);
		//
		
		blockSize = VOIGT_BLOCK;
		gridSize = Npoints;
		//cudaDeviceSetSharedMemConfig() 
		cudaFuncSetCacheConfig(device_compute_voigt, cudaFuncCachePreferL1);
		device_compute_voigt<<<gridSize,blockSize,0,stream>>>(g_freq, g_intens,g_nu,g_abs,g_gamma,sqrt(temp),press,LORENTZ_CUTOFF,Npoints,N_ener,start_idx);
		
}


