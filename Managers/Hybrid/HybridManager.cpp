#include "HybridManager.h"
#include <climits>


HybridManager::HybridManager(ProfileType pProfile,int num_threads,size_t total_memory) : BaseManager(pProfile), selected_worker(0){

    	//Alocated all the managers needed to do crossection
	//workers new BaseManager*[t_num_workers];
	printf("num_threads=%i\n",num_threads);

	#ifdef GPU_ENABLED
	workers.push_back(new MultiGpuManager(pProfile));

	#endif
	if(num_threads>0)
		workers.push_back(new OpenMPManager(pProfile,num_threads,total_memory));
	t_num_workers = workers.size();
	current_worker = workers[selected_worker];
}	




HybridManager::~HybridManager(){
	Cleanup();
}
void HybridManager::InitializeVectors(int Npoints){
	N_trans = INT_MAX;
	for(int i = 0; i < t_num_workers; i++){
		workers[i]->InitializeVectors(Npoints);
		// get the largest NTrans
	}
	for(int i = 0; i < t_num_workers; i++)
		N_trans = std::min(workers[i]->GetNtrans(),N_trans); 
	printf("HybridManager: N_trans = %d\n",N_trans);

}

void HybridManager::InitializeVectors(int Npoints,int num_cs){
	num_intens = num_cs;
	N_trans = INT_MAX;
	if(num_cs==1){
		InitializeVectors(Npoints);
		return;
	}
	for(int i = 0; i < t_num_workers; i++){
		workers[i]->InitializeVectors(Npoints,num_cs);
		// get the largest NTrans
	}
	for(int i = 0; i < t_num_workers; i++)
		N_trans = std::min(workers[i]->GetNtrans(),N_trans); 
	printf("HybridManager: N_trans = %d\n",N_trans);

}

void HybridManager::InitializeConstants(double half_width,double temperature, double partition,double dfreq,double meanmass,double pressure,double ref_temp){

	for(int i = 0; i < t_num_workers; i++){
		workers[i]->InitializeConstants(half_width,temperature, partition,dfreq,meanmass,pressure,ref_temp);
	}


}
void HybridManager::TransferVectors(size_t Nener,double* h_energies, double* h_nu, double* h_aif,int* h_gns,double* h_gamma,double * h_n){
	SelectWorker();
	current_worker->TransferVectors(Nener,h_energies, h_nu, h_aif,h_gns,h_gamma,h_n);

}
void HybridManager::TransferFreq(double* h_freq,double* h_intens,int N){
	for(int i = 0; i < t_num_workers; i++){
		workers[i]->TransferFreq( h_freq,h_intens,N);
	}	

}
void HybridManager::ExecuteCrossSection(int N, int N_ener,int start_idx){
	printf("Worker %d is execturing cross-section\n",selected_worker);
	current_worker->ExecuteCrossSection(N,N_ener,start_idx);
}

void HybridManager::ExecuteCrossSection(int N, int N_ener,int start_idx,double temperature,double pressure,double partition){
	printf("Worker %d is execturing cross-section\n",selected_worker);
	current_worker->ExecuteCrossSection(N,N_ener,start_idx,temperature,pressure,partition);
}

void HybridManager::TransferResults(double* h_freq,double* h_intens,int N){
	double* temp_freq = new double[N];
	double* temp_intens = new double[N];
	for(int i = 0; i < N; i++){
		temp_intens[i] = 0.0;
	}	
	printf("Transferring results from all workers into host!\n");
	
	for(int i = 0; i < t_num_workers; i++){
		workers[i]->TransferResults(temp_freq,temp_intens,N);
		for(int i = 0; i < N; i++){
			h_intens[i] += temp_intens[i];
		}			


	}

	delete[] temp_freq;
	delete[] temp_intens;
	


}

void HybridManager::TransferResults(double* h_freq,double* h_intens,int N,int I){

	if(num_intens==1){
		TransferResults(h_freq,h_intens,N);
		return;
	}

	double* temp_freq = new double[N];
	double* temp_intens = new double[N];
	for(int i = 0; i < N; i++){
		temp_intens[i] = 0.0;
	}	
	printf("Transferring results from all workers into host!\n");
	
	for(int i = 0; i < t_num_workers; i++){
		workers[i]->TransferResults(temp_freq,temp_intens,N,I);
		for(int i = 0; i < N; i++){
			h_intens[i] += temp_intens[i];
		}			


	}

	delete[] temp_freq;
	delete[] temp_intens;
	


}


void HybridManager::Cleanup(){
	for(int i = 0; i < t_num_workers; i++){
		workers[i]->Cleanup();
	}	
	workers.clear();

}

bool HybridManager::ReadyForWork(){
	bool status = false;
	for(int i = 0; i < t_num_workers; i++){
		status |= workers[i]->ReadyForWork();
	}
	return status;

}


void HybridManager::SelectWorker(){
	int i = 0;	
	//Wait until we have a worker
	while(true){
		
   		if(i >= t_num_workers)
			i=0;
		
		if(workers[i]->ReadyForWork()){
			selected_worker=i;
			current_worker = workers[i];
			break;
		}
		i++;
	}		

}

