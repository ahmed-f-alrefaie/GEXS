#include "BaseManager.h"


void BaseManager::InitializeMemory(size_t bytes){
	available_memory = bytes;
	total_memory = bytes;
}

void BaseManager::TrackMemory(size_t bytes){
	
	available_memory -= bytes;
}
void BaseManager::FreeMemory(size_t bytes){
	available_memory += bytes;
}
size_t BaseManager::GetAvailableMemory(){
	return available_memory;
}
BaseManager::BaseManager(ProfileType pprofile) : profile(pprofile){}

BaseManager::~BaseManager(){}

size_t BaseManager::GetNtrans(){return N_trans;}

bool BaseManager::ReadyForWork(){
	return true;
}

