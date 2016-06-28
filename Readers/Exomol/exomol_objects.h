#pragma once

struct state{
	int id;
	double energy;
	int gns;
	int J;
};

struct exomol_states{
	state* states;
	int N;
	double partition;
};


	
	

