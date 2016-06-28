#ifndef DEFINES_H_
#define DEFINES_H_

#define LN2 0.69314718056
#define LN2PI 0.22063560015
#define SQRTLN2PI  0.46971863935
#define ISQRTLN2 1.201122409
#define SQRTLN2 0.832554611
#define ISQRTPI 0.564189584
#define SQRTPI 1.772453851
#define IPI 0.318309886

#define NA 6.0221412927e23
#define DOPPLER_CUTOFF 100

#define MAX_QUADS 1


#define CM_COEF 1.327209365e-12
#define PI 3.14159265359
#define PLANCK 6.6260693e-27
#define VELLGT 2.99792458e+10
#define BOLTZ 1.380658e-16
#define BETA_PREF 1.438767314
#define LORENTZ_CUTOFF 100

//#define GPU_ENABLED
#define MAX_STREAMS 32

#ifdef KEPLER
#define BLOCK_SIZE 1024

#else
#define BLOCK_SIZE 576
#endif
#define SHARED_SIZE 128
#define DOPPLER_SIZE 128
#define VOIGT_BLOCK 128
#define VOIGT_SHARED_SIZE 128
#define VOIGT_QUAD_BLOCK 128
#define WARP_SIZE 32

#endif