#include "GPUManager.h"
#include "Timer.h"
#include "cuda_utils.cuh"

GpuManager::GpuManager(ProfileType pProfile,int pgpu_id) : BaseManager(pProfile), alloc(false),g_freq(NULL),g_intens(NULL),g_energies(NULL),g_nu(NULL),g_aif(NULL),g_gns(NULL),g_gamma(NULL),g_n(NULL), gpu_id(pgpu_id){
	cudaSetDevice(gpu_id);
	cudaFree(0);
	//Get device properties
	cudaDeviceProp devProp;
        cudaGetDeviceProperties(&devProp, 0);
	//Now we compute total bytes
	InitializeMemory((devProp.totalGlobalMem*0.95));
	printf("Available memory on GPU: %d  %zu bytes\n",gpu_id,GetAvailableMemory());
	//available_memory /=6l;	
	CheckCudaError("Get Device Properties");
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

}
GpuManager::~GpuManager(){
	Cleanup();
	
}


void GpuManager::InitializeConstants(double half_width,double temperature, double partition,double dfreq,double mean_mass,double pressure,double ref_temp){
	cudaSetDevice(gpu_id);
	cross_section_data cross_constants;

	double planck =6.6260693e-27;
	double vellgt=2.99792458e10;
	double boltz=1.380658e-16;
	double pi=std::acos(-1.0);
	double ln2 = std::log(2.0);
	double avogno = 6.02214129e+23;
	double dpwco = sqrt(2.0*ln2*boltz*avogno/mean_mass)/vellgt;
	printf ("HW = %12.6f temp=%12.6f partition=%12.6f\n",half_width,temperature, partition);
	printf("Doppler constant = %14.4E\n",dpwco);
	cross_constants.dpwcoeff = dpwco;
	cross_constants.halfwidth = half_width;
	cross_constants.beta = planck*vellgt/(boltz*temperature);
	cross_constants.partition = partition;
	cross_constants.ref_temp= ref_temp;
	//cross_constants.ln2pi= sqrt(cross_constants.ln2/pi);
	cross_constants.cmcoef=1.0/(8.0*pi*vellgt);
	cross_constants.dfreq = dfreq;
	cross_constants.temperature = temperature;
	cross_constants.mean_mass = mean_mass;
	cross_constants.pressure = pressure;
	copy_intensity_info(&cross_constants);
	CheckCudaError("Copy Constants");

}

void GpuManager::InitializeVectors(int Npoints){
	cudaSetDevice(gpu_id);
	cudaMalloc((void**)&g_freq,sizeof(double)*size_t(Npoints));
	TrackMemory(sizeof(double)*size_t(Npoints));
	cudaMalloc((void**)&g_intens,sizeof(double)*size_t(Npoints));
	TrackMemory(sizeof(double)*size_t(Npoints));
	if(profile==VOIGT)
		N_trans = GetAvailableMemory()/(sizeof(double)+sizeof(double)+sizeof(double)+sizeof(int)+sizeof(double) + sizeof(double));
	else
		N_trans = GetAvailableMemory()/(sizeof(double)+sizeof(double)+sizeof(double)+sizeof(int));
	printf("Number of transitions in the GPU = %d\n",N_trans);
	cudaMalloc((void**)&g_energies,sizeof(double)*size_t(N_trans));
	TrackMemory(sizeof(double)*size_t(N_trans));
	cudaMalloc((void**)&g_nu,sizeof(double)*size_t(N_trans));
	TrackMemory(sizeof(double)*size_t(N_trans));
	cudaMalloc((void**)&g_aif,sizeof(double)*size_t(N_trans));
	TrackMemory(sizeof(double)*size_t(N_trans));
	cudaMalloc((void**)&g_gns,sizeof(int)*size_t(N_trans));
	TrackMemory(sizeof(int)*size_t(N_trans));
	if(profile==VOIGT){
		//printf("Allocating Voigt stuff\n");
		cudaMalloc((void**)&g_gamma,sizeof(double)*size_t(N_trans));
		TrackMemory(sizeof(double)*size_t(N_trans));
		cudaMalloc((void**)&g_n,sizeof(double)*size_t(N_trans));
		TrackMemory(sizeof(double)*size_t(N_trans));
	}
	CheckCudaError("Initialize Vectors");
}

void GpuManager::InitializeVectors(int Npoints,int num_cs){
	cudaSetDevice(gpu_id);
	//if(num_cs == 1){
	//	InitializeVectors(Npoints);
	//	return;
	//}
		
	cudaMalloc((void**)&g_freq,sizeof(double)*size_t(Npoints));
	TrackMemory(sizeof(double)*size_t(Npoints));
	for(int i = 0; i < num_cs; i++){
		g_tp_intens.push_back(NULL);
		cudaMalloc((void**)&g_tp_intens.back(),sizeof(double)*size_t(Npoints));
		TrackMemory(sizeof(double)*size_t(Npoints));
	}
	if(profile==VOIGT)
		N_trans = GetAvailableMemory()/(sizeof(double)+sizeof(double)+sizeof(double)+size_t(num_cs)*sizeof(double)+sizeof(int)+sizeof(double) + sizeof(double));
	else
		N_trans = GetAvailableMemory()/(sizeof(double)+sizeof(double)+sizeof(double)+size_t(num_cs)*sizeof(double)+sizeof(int));
	printf("Number of transitions in the GPU = %d\n",N_trans);

	cudaMalloc((void**)&g_energies,sizeof(double)*size_t(N_trans));
	TrackMemory(sizeof(double)*size_t(N_trans));

	cudaMalloc((void**)&g_nu,sizeof(double)*size_t(N_trans));
	TrackMemory(sizeof(double)*size_t(N_trans));

	cudaMalloc((void**)&g_aif,sizeof(double)*size_t(N_trans));
	TrackMemory(sizeof(double)*size_t(N_trans));

	for(int i = 0; i < num_cs; i++){
		g_tp_stream.push_back(NULL);
		g_tp_abscoef.push_back(NULL);
		cudaStreamCreate(&g_tp_stream.back());
		cudaMalloc((void**)&g_tp_abscoef.back(),sizeof(double)*size_t(N_trans));
		TrackMemory(sizeof(double)*size_t(N_trans));
	}


	cudaMalloc((void**)&g_gns,sizeof(int)*size_t(N_trans));
	TrackMemory(sizeof(int)*size_t(N_trans));
	if(profile==VOIGT){
		//printf("Allocating Voigt stuff\n");
		cudaMalloc((void**)&g_gamma,sizeof(double)*size_t(N_trans));
		TrackMemory(sizeof(double)*size_t(N_trans));
		cudaMalloc((void**)&g_n,sizeof(double)*size_t(N_trans));
		TrackMemory(sizeof(double)*size_t(N_trans));
	}
	CheckCudaError("Initialize Vectors");
}

void GpuManager::TransferFreq(double* h_freq,double* h_intens,int N){
	cudaSetDevice(gpu_id);
	cudaMemcpy(g_freq,h_freq,sizeof(double)*size_t(N),cudaMemcpyHostToDevice);
	if(g_tp_intens.size()==0)
		cudaMemcpy(g_intens,h_intens,sizeof(double)*size_t(N),cudaMemcpyHostToDevice);
	else{
		for(int i =0; i < g_tp_intens.size(); i++)
			cudaMemcpy(g_tp_intens[i],h_intens,sizeof(double)*size_t(N),cudaMemcpyHostToDevice);
	}
	cudaDeviceSynchronize();
	CheckCudaError("Copy");
}
void GpuManager::TransferVectors(size_t Nener,double* h_energies, double* h_nu, double* h_aif,int* h_gns,double* h_gamma,double * h_n){
	cudaSetDevice(gpu_id);
	//We reset tasks 
	tasks = 0;
	cudaDeviceSynchronize(); // Synchronize at the beginning
	CheckCudaError("Kernal");
	Timer::getInstance().StartTimer("MemCpy");
	//printf("Performing Copy, Ntransitions = %d\n",Nener);
	cudaMemcpy(g_energies,h_energies,sizeof(double)*size_t(Nener),cudaMemcpyHostToDevice);
	cudaMemcpy(g_nu,h_nu,sizeof(double)*size_t(Nener),cudaMemcpyHostToDevice);
	cudaMemcpy(g_aif,h_aif,sizeof(double)*size_t(Nener),cudaMemcpyHostToDevice);
	cudaMemcpy(g_gns,h_gns,sizeof(int)*size_t(Nener),cudaMemcpyHostToDevice);
	if(h_gamma != NULL && h_n != NULL){
	//	printf("Transferred gammas\n");
		cudaMemcpy(g_gamma,h_gamma,sizeof(double)*size_t(Nener),cudaMemcpyHostToDevice);
		cudaMemcpy(g_n,h_n,sizeof(double)*size_t(Nener),cudaMemcpyHostToDevice);
	}
	cudaDeviceSynchronize();
	CheckCudaError("Copy");
	Timer::getInstance().EndTimer("MemCpy");
}

void GpuManager::ExecuteCrossSection(int N, int N_ener,int start_idx){
	cudaSetDevice(gpu_id);
	printf("Executing cross-section on GPU %d\n",gpu_id);
	/*if(profile == GAUSSIAN)
		//execute_two_step_kernal_block(g_freq, g_intens,g_energies,g_nu,g_gns,g_aif,N,N_ener,start_idx);
	else if(profile == DOPPLER)
		//printf("Not implemented\n");
		ExecuteDopplerCrossSection(N, N_ener,start_idx);
	else if(profile == VOIGT){
		//if(N > 100000)
		//	ExecuteVoigtCrossSection(N, N_ener,start_idx);
		//else
			ExecuteVoigtCrossSectionBlock(N, N_ener,start_idx);
	}*/
	printf("Depreceated function");
	exit(0);
}

void GpuManager::ExecuteCrossSection(int N, int N_ener,int start_idx,double temperature,double pressure,double partition){
	cudaSetDevice(gpu_id);
	//printf("Executing cross-section on GPU %d\n",gpu_id);
	if(profile == GAUSSIAN)
		ExecuteGaussianCrossSection(N, N_ener,start_idx,temperature,partition,pressure);
	else if(profile == DOPPLER)
		ExecuteDopplerCrossSection(N, N_ener,start_idx,temperature,partition);
		//printf("Not implemented\n");
		//printf("Not implemented\n");
	else if(profile == VOIGT){
		//if(N > 100000)
		//	ExecuteVoigtCrossSection(N, N_ener,start_idx);
		//else
			ExecuteVoigtCrossSectionBlock(N, N_ener,start_idx,temperature,pressure,partition);
	}
}


void GpuManager::ExecuteGaussianCrossSection(int N, int N_ener,int start_idx){
		cudaSetDevice(gpu_id);		
		//execute_two_step_kernal_block(g_freq, g_intens,g_energies,g_nu,g_gns,g_aif,N,N_ener,start_idx);
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
		//cudaDeviceSynchronize();
		//Timer::getInstance().EndTimer("Kernal Execution");	
}

void GpuManager::ExecuteGaussianCrossSection(int N, int N_ener,int start_idx,double temperature,double partition,double hw){
		cudaSetDevice(gpu_id);		
		gpu_compute_lorentzian_profile(g_freq, g_tp_intens[tasks],g_energies,g_nu,g_gns,g_aif,g_tp_abscoef[tasks],temperature,partition,hw,N,N_ener,start_idx,g_tp_stream[tasks]);
		tasks++;
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
		//cudaDeviceSynchronize();
		//Timer::getInstance().EndTimer("Kernal Execution");	
}

void GpuManager::ExecuteDopplerCrossSection(int N, int N_ener,int start_idx){
		cudaSetDevice(gpu_id);
		//execute_two_step_kernal_doppler_block(g_freq, g_intens,g_energies,g_nu,g_gns,g_aif,N,N_ener,start_idx);
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
		//cudaDeviceSynchronize();
		//Timer::getInstance().EndTimer("Kernal Execution");	
}
void GpuManager::ExecuteDopplerCrossSection(int N, int N_ener,int start_idx,double temperature,double partition){
		cudaSetDevice(gpu_id);
		//printf("Task %d\n",tasks);
		//if(g_gamma==NULL)
		//	execute_two_step_kernal_voigt_block(g_freq, g_intens,g_energies,g_nu,g_gns,g_aif,N,N_ener,start_idx);
		//else
		gpu_compute_doppler_profile(g_freq, g_tp_intens[tasks],g_energies,g_nu,g_gns,g_aif,g_tp_abscoef[tasks],temperature,partition,N,N_ener,start_idx,g_tp_stream[tasks]);
		tasks++;
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
		//cudaDeviceSynchronize();
		//Timer::getInstance().EndTimer("Kernal Execution");	
}

void GpuManager::ExecuteVoigtCrossSectionBlock(int N, int N_ener,int start_idx){
		cudaSetDevice(gpu_id);
		//if(g_gamma==NULL)
		//	execute_two_step_kernal_voigt_block(g_freq, g_intens,g_energies,g_nu,g_gns,g_aif,N,N_ener,start_idx);
		//else
			//execute_two_step_kernal_voigt_block(g_freq, g_intens,g_energies,g_nu,g_gns,g_aif,g_gamma,g_n,N,N_ener,start_idx);
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
		//cudaDeviceSynchronize();
		//Timer::getInstance().EndTimer("Kernal Execution");	
}

void GpuManager::ExecuteVoigtCrossSectionBlock(int N, int N_ener,int start_idx,double temperature,double pressure,double partition){
		cudaSetDevice(gpu_id);
		//printf("Task %d\n",tasks);
		gpu_compute_voigt_profile(g_freq, g_tp_intens[tasks],g_energies,g_nu,g_gns,g_aif,g_tp_abscoef[tasks],g_gamma,g_n,temperature,pressure,partition,N,N_ener,start_idx,g_tp_stream[tasks]);
		tasks++;
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
		//cudaDeviceSynchronize();
		//Timer::getInstance().EndTimer("Kernal Execution");	
}

void GpuManager::ExecuteVoigtCrossSection(int N, int N_ener,int start_idx){
		cudaSetDevice(gpu_id);		
		//execute_two_step_kernal_voigt(g_freq, g_intens,g_energies,g_nu,g_gns,g_aif,N,N_ener,start_idx);
					//device_compute_cross_section_abscoef<<<gridSize,blockSize>>>(g_freq, g_cs,g_energies,g_gns,g_nu,g_aif,Npoints,N_ener,0);
		//cudaDeviceSynchronize();
		//Timer::getInstance().EndTimer("Kernal Execution");	
}


void GpuManager::TransferResults(double* h_freq,double* h_intens,int N){
	cudaSetDevice(gpu_id);
	cudaDeviceSynchronize();
	cudaMemcpy(h_freq,g_freq,sizeof(double)*size_t(N),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_intens,g_intens,sizeof(double)*size_t(N),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	CheckCudaError("Copy Results");
}

void GpuManager::TransferResults(double* h_freq,double* h_intens,int N, int I){
	cudaSetDevice(gpu_id);
	cudaDeviceSynchronize();
	cudaMemcpy(h_freq,g_freq,sizeof(double)*size_t(N),cudaMemcpyDeviceToHost);
	cudaMemcpy(h_intens,g_tp_intens[I],sizeof(double)*size_t(N),cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	CheckCudaError("Copy Results");
}


void GpuManager::Cleanup(){
	cudaSetDevice(gpu_id);
	cudaFree(g_freq);
	cudaFree(g_intens);
	cudaFree(g_energies);
	cudaFree(g_gns);
	cudaFree(g_aif);
	cudaFree(g_gamma);
	cudaFree(g_n);


}	

bool GpuManager::ReadyForWork(){
	cudaSetDevice(gpu_id);
	if(g_tp_intens.size() == tasks || g_tp_intens.size()==0)
		return cudaSuccess==cudaStreamQuery(0);
	else
		return true;
}

