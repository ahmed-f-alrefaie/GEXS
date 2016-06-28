#ifndef CROSS_KERNAL__V2_H
#define CROSS_KERNAL_H

#include "cross_structs.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>

//----------------------HOST-FUNCTIONS--------------------------------
//Compute absolute intensities
//__host__ void device_compute_abscoefs(const double* g_energies,const int*  g_gns,const double*  g_nu,const double*  g_aif,double* g_abscoef,const double temperature,const double partition, const int N_ener);
//Compute pressure 
//__host__ void device_compute_pressure(double*  g_gamma,double*  g_n,double temperature);
__host__ void copy_intensity_info(cross_section_data* cross_inf);

//Lorentzian
__host__ void gpu_compute_lorentzian_profile(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double temp,double part,double hw, int Npoints,int N_ener,int start_idx,cudaStream_t stream);
//Gaussian
__host__ void gpu_compute_gaussian_profile(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double temp,double part,double hw, int Npoints,int N_ener,int start_idx,cudaStream_t stream);
//Doppler
__host__ void gpu_compute_doppler_profile(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double temp,double part, int Npoints,int N_ener,int start_idx,cudaStream_t stream);
//Voigt
__host__ void gpu_compute_voigt_profile(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double* g_gamma,double* g_n ,double temp,double press,double part,int Npoints,int N_ener,int start_idx,cudaStream_t stream);


#endif
