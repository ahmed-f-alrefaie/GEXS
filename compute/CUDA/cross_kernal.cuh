#ifndef CROSS_KERNAL_H
#define CROSS_KERNAL_H

#include "cross_structs.cuh"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cmath>




__host__ void copy_intensity_info(cross_section_data* cross_inf);

//__global__ void device_compute_cross_section(const double* g_freq, double* g_cs,const double* g_energies,const int* g_gns,const int* g_jF,const double* g_nu,const double* g_aif, const int N,const int N_ener,const int start_idx);

__global__ void device_compute_cross_section(const double*  __restrict__ g_freq, double* g_cs,const double*  __restrict__ g_energies,const int*  __restrict__ g_gns,const int*  __restrict__ g_jF,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N,const int N_ener,const int start_idx);

__global__ void device_compute_cross_section_abscoef(const double*  __restrict__ g_freq, double* g_cs,const double*  __restrict__ g_energies,const int*  __restrict__ g_gns,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N,const int N_ener,const int start_idx);

__global__ void device_compute_cross_section_warp_abscoef(const double*  __restrict__ g_freq, double* g_cs,const double*  __restrict__ g_energies,const int*  __restrict__ g_gns,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N,const int N_ener,const int start_idx);

__global__ void device_compute_cross_section_stepone(double* g_energies,const int*  __restrict__ g_gns,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N_ener);
__global__ void device_compute_cross_section_steptwo(const double*  __restrict__ g_freq, double* g_cs,const double*  __restrict__ g_nu,const double*  __restrict__ g_abscoef,const int N,const int N_ener,const int start_idx);

__host__ void execute_two_step_kernal(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif, int Npoints,int N_ener);

__global__ void device_compute_cross_section_steptwo_block(const double* g_freq, double* g_cs,const double* g_nu,const double* g_abscoef,const int N,const int N_ener,const int start_idx);

__global__ void device_compute_cross_section_steptwo_block_hw(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,double hw,const int N,const int N_ener,const int start_idx);

__host__ void execute_two_step_kernal_block(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif, int Npoints,int N_ener,int start_idx);

__host__ void execute_two_step_kernal_block_temp(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double temp,double part,double hw, int Npoints,int N_ener,int start_idx,cudaStream_t stream);

//VOIGT

__global__ void device_compute_cross_section_voigt_stepone(double* g_energies,const int*  __restrict__ g_gns,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N_ener);
__global__ void device_compute_cross_section_voigt_stepone(double* g_energies,const int*  g_gns,const double*  g_nu,const double*  g_aif,double*  g_gamma,double*  g_n, const int N_ener);
__global__ void device_compute_cross_section_voigt_steptwo_block(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double*  g_gamma,const int N,const int N_ener,const int start_idx);
//__global__ void device_compute_cross_section_voigt_steptwo_block(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const int N,const int N_ener,const int start_idx);

__global__ void device_compute_cross_section_voigt_single(const double* g_freq,double* g_cs,const double   nu_if,const double ei, const double gns,const double  aif,const double  g_gamma, const double pressure,const double p_n, const double temperature, const double partition, const int N);

__host__ void execute_two_step_kernal_voigt_block(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif, int Npoints,int N_ener,int start_idx);
__host__ void execute_two_step_kernal_voigt_block(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_gamma,double* g_n ,int Npoints,int N_ener,int start_idx);
__host__ void execute_two_step_kernal_voigt(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif, int Npoints,int N_ener,const int start_idx);

__global__ void device_compute_cross_section_voigt_steptwo(const double* g_freq, double* g_cs,const double*  g_nu,const double*  g_abscoef,const int N,const int N_ener,const int start_idx);

__global__ void device_compute_cross_section_voigt_quad_steptwo_block(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double*  g_gamma,const int N,const int N_ener,const int start_idx);

__host__ void execute_two_step_kernal_voigt_quad_block(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_gamma,double* g_n ,int Npoints,int N_ener,int start_idx);
__host__ void execute_two_step_kernal_voigt_single(double* g_freq,double* g_cs,const double   nu_if,const double ei, const double gns,const double  aif,const double  g_gamma, const double pressure,const double p_n, const double temperature, const double partition, const int N, cudaStream_t stream);

//TEMP PRESSURE


__global__ void device_compute_cross_section_voigt_stepone_temp_press(const double* g_energies,const int*  g_gns,const double*  g_nu,const double*  g_aif,double*  g_abscoef,double*  g_gamma,double*  g_n,double temperature, double pressure,double partition, const int N_ener);

__global__ void device_compute_cross_section_voigt_steptwo_block_temp_press(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double*  g_gamma,const double temperature,const double pressure,const int N,const int N_ener,const int start_idx);

__host__ void execute_two_step_kernal_voigt_block_temp_press(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double* g_gamma,double* g_n ,double temp,double press,double part,int Npoints,int N_ener,int start_idx,cudaStream_t stream);

//////DOPPLER////////////
__host__ void execute_two_step_kernal_doppler_block(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif, int Npoints,int N_ener,int start_idx);
__global__ void device_compute_cross_section_doppler_stepone(double* g_energies,const int*  __restrict__ g_gns,const double*  __restrict__ g_nu,const double*  __restrict__ g_aif, const int N_ener);
__global__ void device_compute_cross_section_doppler_steptwo_block(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const int N,const int N_ener,const int start_idx);

//TEMP-PRESSURE
__global__ void device_compute_cross_section_doppler_stepone_temp(const double* g_energies,const int*   g_gns,const double*   g_nu,const double*  g_aif,double*  g_abscoef,double temperature,double partition, const int N_ener);
__global__ void device_compute_cross_section_doppler_steptwo_block_temp(const double*  g_freq, double* g_cs,const double*   g_nu,const double*  g_abscoef,const double temperature,const int N,const int N_ener,const int start_idx);
__host__ void execute_two_step_kernal_doppler_block_temp(double* g_freq, double* g_intens, double* g_energies, double* g_nu, int* g_gns,double* g_aif,double* g_abs,double temp,double part, int Npoints,int N_ener,int start_idx,cudaStream_t stream);

#endif
