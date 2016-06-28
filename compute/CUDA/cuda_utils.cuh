#include <cuda_runtime_api.h>
#include <cuda.h>
#include <cstdio>
#pragma once


void CheckCudaError(const char* tag);

// Print device properties
void printDevProp(cudaDeviceProp devProp);

