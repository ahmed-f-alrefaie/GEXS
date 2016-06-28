#include "exomol_objects.h"
#include "exomol_functions.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

void initialize_states(const char* filename, exomol_states* states){
	int _N=0;
	FILE* state_file = fopen(filename,"r");
	int tmp_N = 0;

	char buffer[1024];
	
	while(fgets(buffer, 1024, state_file)) {
   		// printf("%s\n", buffer);
		 tmp_N++;
	}

	printf("Number of energies = %d\n",tmp_N);
	states->N = tmp_N;
	
	rewind(state_file);
	states->states = new state[tmp_N];
	
	char* ln_ptr;
	tmp_N = 0;
	while(fgets(buffer, 1024, state_file)) {
		states->states[tmp_N].id=strtol(buffer,&ln_ptr,0)-1;
		//Energy
		states->states[tmp_N].energy=strtod(ln_ptr,&ln_ptr);
		//Gns
		states->states[tmp_N].gns=strtol(ln_ptr,&ln_ptr,0);
		//J
		states->states[tmp_N].J=strtol(ln_ptr,&ln_ptr,0);
		

		//printf("I = %d  E = %12.6f GNS = %d  J = %d \n",states->states[tmp_N].id,states->states[tmp_N].energy,
		//states->states[tmp_N].gns,states->states[tmp_N].J);

		tmp_N++;
	}	

	fclose(state_file);



}

double compute_partition( exomol_states* states,double temp){
	int n = states->N;

	double partition=0.0;
	double temperature = 0.0;
	double planck =6.6260693e-27;
	double avogno=6.0221415e23;
	double vellgt=2.99792458e10;
	double boltz=1.380658e-16;
	double pi=std::acos(-1.0);
	double beta=planck*vellgt/(boltz*temp);

	//printf("Computing the partition for %d states\n",n);
	for(int i = 0; i < n; i ++){

		partition+= states->states[i].gns*exp(-beta*states->states[i].energy);

	}
	states->partition = partition;
	printf("Partition function = %12.6f\n",partition);
	
	return partition;

}
