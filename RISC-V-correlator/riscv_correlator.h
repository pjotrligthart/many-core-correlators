#ifndef _RISCV_CORRELATOR_H
#define _RISCV_CORRELATOR_H

#include "../common/common.h"

#define VECTOR_WIDTH_IN_FLOATS 8
#define ALIGNMENT 256 // in bytes

#define CORRELATOR_1X1_RVV      5
#define CORRELATOR_1X1_TIME_RVV 6
#define CORRELATOR_2X2_RVV      7
#define CORRELATOR_2X2_TIME_RVV 8
#define CORRELATOR_3X2_TIME_RVV 9
#define CORRELATOR_1X1_TIME_RVV128 10


extern unsigned long long cpuCorrelator_1x1_rvv(const float* __restrict__ samples, float* __restrict__ visibilities,
						 unsigned nrTimes, const unsigned nrTimesWidth,
						 const unsigned nrStations, const unsigned nrChannels,
						 unsigned long long* bytesLoaded, unsigned long long* bytesStored);

extern unsigned long long cpuCorrelator_1x1_time_rvv(const float* __restrict__ samples, float* __restrict__ visibilities,
						      const unsigned nrTimes, const unsigned nrTimesWidth,
						      const unsigned nrStations, const unsigned nrChannels,
						      unsigned long long* bytesLoaded, unsigned long long* bytesStored);

extern unsigned long long cpuCorrelator_2x2_rvv(const float* __restrict__ samples, float* __restrict__ visibilities,
						 const unsigned nrTimes, const unsigned nrTimesWidth,
						 const unsigned nrStations, const unsigned nrChannels,
						 unsigned long long* bytesLoaded, unsigned long long* bytesStored);

extern unsigned long long cpuCorrelator_2x2_time_rvv(const float* __restrict__ samples, float* __restrict__ visibilities,
						      const unsigned nrTimes, const unsigned nrTimesWidth,
						      const unsigned nrStations, const unsigned nrChannels,
						      unsigned long long* bytesLoaded, unsigned long long* bytesStored);

extern unsigned long long cpuCorrelator_3x2_time_rvv(const float* __restrict__ samples, float* __restrict__ visibilities,
						      const unsigned nrTimes, const unsigned nrTimesWidth,
						      const unsigned nrStations, const unsigned nrChannels,
						      unsigned long long* bytesLoaded, unsigned long long* bytesStored);

extern unsigned long long cpuCorrelator_1x1_time_rvv128(const float* __restrict__ samples, float* __restrict__ visibilities,
						      const unsigned nrTimes, const unsigned nrTimesWidth,
						      const unsigned nrStations, const unsigned nrChannels,
						      unsigned long long* bytesLoaded, unsigned long long* bytesStored);

#endif // _RISCV_CORRELATOR_H
