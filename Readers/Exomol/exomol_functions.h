#include "exomol_objects.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

#pragma once

void initialize_states(const char* filename, exomol_states* states);
double compute_partition( exomol_states* states,double temp);
