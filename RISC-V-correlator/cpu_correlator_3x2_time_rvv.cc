#include <cstdlib>
#include <iostream>
#include <riscv_vector.h> // rvv

#include "riscv_correlator.h"


using namespace std;

static unsigned char cellToStatX[MAX_NR_CELLS], cellToStatY[MAX_NR_CELLS];

static unsigned fillCellToStatTable(const unsigned nrStations)
{
    unsigned nrCells, stat0, stat2;

    for (stat2 = nrStations % 2 ? 1 : 2, nrCells = 0; stat2 < nrStations; stat2 += 2) {
	for (stat0 = 0; stat0 + 2 <= stat2; stat0 += 3, nrCells ++) {
	    cellToStatX[nrCells] = stat0;
	    cellToStatY[nrCells] = stat2;
	}
    }

    return nrCells;
}

static unsigned long long calcNrOps(const unsigned nrTimes, const unsigned nrStations, const unsigned nrChannels,
				    unsigned long long* bytesLoaded, unsigned long long* bytesStored)
{
    unsigned nrCells, stat0, stat2;

    for (stat2 = nrStations % 2 ? 1 : 2, nrCells = 0; stat2 < nrStations; stat2 += 2) {
	for (stat0 = 0; stat0 + 2 <= stat2; stat0 += 3, nrCells ++);
    }

    unsigned long long ops = (unsigned long long)nrChannels * nrCells * nrTimes * 32L * 6L;

    *bytesLoaded = (unsigned long long)nrChannels * nrCells * nrTimes * 4L * 4L * 5L;
    *bytesStored = (unsigned long long)nrChannels * nrCells * 2L * 4L * 5L;

    return ops;
}

inline vfloat32m8_t i_vfloat32m8(const float val)
{
    float result[VECTOR_WIDTH_IN_FLOATS];
    for(int i = 0; i < VECTOR_WIDTH_IN_FLOATS; i++) {
	result[i] = val;
    }

    return __riscv_vle32_v_f32m8(result, VECTOR_WIDTH_IN_FLOATS);
}

// (Temporary?) function that reduces/sums all values in a vfloat32m8 to a single float
inline float reduce_vfloat32m8(const vfloat32m8_t &src)
{
    vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    vfloat32m1_t res = __riscv_vfmv_s_f_f32m1(0.0f, 1);

    // Use vfredusum because it seems to be faster on our target board (BPI-F3)
    // See benchmarks at https://camel-cdr.github.io/rvv-bench-results/bpi_f3/index.html
    res = __riscv_vfredusum_vs_f32m8_f32m1(src, zero, 8);
    return __riscv_vfmv_f_s_f32m1_f32(res);
}

unsigned long long cpuCorrelator_3x2_time_rvv(const float* __restrict__ samples, float* __restrict__ visibilities, 
						      const unsigned nrTimes, const unsigned nrTimesWidth, 
						      const unsigned nrStations, const unsigned nrChannels,
						      unsigned long long* bytesLoaded, unsigned long long* bytesStored)
{
    const unsigned nrBaselines = NR_BASELINES(nrStations);
    bool missedBaselines[nrBaselines*2]; // we sometimes go beyond the #baslines, but then we don't store the visibilities, so that is ok.
    for(unsigned baseline=0; baseline < nrBaselines; baseline++) {
	missedBaselines[baseline] = true;
    }

    unsigned nrCells = fillCellToStatTable(nrStations);

    for (unsigned channel = 0; channel < nrChannels; channel++) {
	for (unsigned cell = 0; cell < nrCells; cell++) {
	    unsigned stat0 = cellToStatX[cell];
	    unsigned stat3 = cellToStatY[cell];
	    size_t index0 = SAMPLE_INDEX(stat0+0, channel, 0, 0, 0);
	    size_t index1 = SAMPLE_INDEX(stat0+1, channel, 0, 0, 0);
	    size_t index2 = SAMPLE_INDEX(stat0+2, channel, 0, 0, 0);
	    size_t index3 = SAMPLE_INDEX(stat3+0, channel, 0, 0, 0);
	    size_t index4 = SAMPLE_INDEX(stat3+1, channel, 0, 0, 0);

	    vfloat32m8_t xxr03 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xxi03 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyr03 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyi03 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxr03 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxi03 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyr03 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyi03 = i_vfloat32m8(0.0f);

	    vfloat32m8_t xxr13 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xxi13 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyr13 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyi13 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxr13 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxi13 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyr13 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyi13 = i_vfloat32m8(0.0f);

	    vfloat32m8_t xxr23 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xxi23 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyr23 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyi23 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxr23 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxi23 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyr23 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyi23 = i_vfloat32m8(0.0f);


	    vfloat32m8_t xxr04 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xxi04 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyr04 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyi04 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxr04 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxi04 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyr04 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyi04 = i_vfloat32m8(0.0f);

	    vfloat32m8_t xxr14 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xxi14 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyr14 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyi14 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxr14 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxi14 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyr14 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyi14 = i_vfloat32m8(0.0f);

	    vfloat32m8_t xxr24 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xxi24 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyr24 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyi24 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxr24 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxi24 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyr24 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyi24 = i_vfloat32m8(0.0f);

	    // assume nrTimes is divisable by vector width
	    for (unsigned time = 0; time < nrTimes; time += VECTOR_WIDTH_IN_FLOATS) {
	      // This assumes a different memory layout with (vector width) samples in time behind each other:
	      // (xr1 xr2 xr3 xr4) (xi1 xi2 xi3 xi4) (yr1 yr2 yr3 yr4) (yi1 yi2 yi3 yi4), etc.
		vfloat32m8_t sample0xr = __riscv_vle32_v_f32m8(samples + time*4 + index0+0*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample0xi = __riscv_vle32_v_f32m8(samples + time*4 + index0+1*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample0yr = __riscv_vle32_v_f32m8(samples + time*4 + index0+2*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample0yi = __riscv_vle32_v_f32m8(samples + time*4 + index0+3*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);

		vfloat32m8_t sample1xr = __riscv_vle32_v_f32m8(samples + time*4 + index1+0*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample1xi = __riscv_vle32_v_f32m8(samples + time*4 + index1+1*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample1yr = __riscv_vle32_v_f32m8(samples + time*4 + index1+2*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample1yi = __riscv_vle32_v_f32m8(samples + time*4 + index1+3*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);

		vfloat32m8_t sample2xr = __riscv_vle32_v_f32m8(samples + time*4 + index2+0*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample2xi = __riscv_vle32_v_f32m8(samples + time*4 + index2+1*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample2yr = __riscv_vle32_v_f32m8(samples + time*4 + index2+2*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample2yi = __riscv_vle32_v_f32m8(samples + time*4 + index2+3*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);

		vfloat32m8_t sample3xr = __riscv_vle32_v_f32m8(samples + time*4 + index3+0*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample3xi = __riscv_vle32_v_f32m8(samples + time*4 + index3+1*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample3yr = __riscv_vle32_v_f32m8(samples + time*4 + index3+2*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample3yi = __riscv_vle32_v_f32m8(samples + time*4 + index3+3*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);

		vfloat32m8_t sample4xr = __riscv_vle32_v_f32m8(samples + time*4 + index4+0*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample4xi = __riscv_vle32_v_f32m8(samples + time*4 + index4+1*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample4yr = __riscv_vle32_v_f32m8(samples + time*4 + index4+2*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample4yi = __riscv_vle32_v_f32m8(samples + time*4 + index4+3*VECTOR_WIDTH_IN_FLOATS, VECTOR_WIDTH_IN_FLOATS);

                xxr03 = __riscv_vfmacc_vv_f32m8(xxr03, sample0xr, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                xxi03 = __riscv_vfmacc_vv_f32m8(xxi03, sample0xi, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                xyr03 = __riscv_vfmacc_vv_f32m8(xyr03, sample0xr, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                xyi03 = __riscv_vfmacc_vv_f32m8(xyi03, sample0xi, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                yxr03 = __riscv_vfmacc_vv_f32m8(yxr03, sample0yr, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                yxi03 = __riscv_vfmacc_vv_f32m8(yxi03, sample0yi, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                yyr03 = __riscv_vfmacc_vv_f32m8(yyr03, sample0yr, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                yyi03 = __riscv_vfmacc_vv_f32m8(yyi03, sample0yi, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                xxr03 = __riscv_vfmacc_vv_f32m8(xxr03, sample0xi, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                xxi03 = __riscv_vfnmsac_vv_f32m8(xxi03, sample0xr, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                xyr03 = __riscv_vfmacc_vv_f32m8(xyr03, sample0xi, sample3yi, VECTOR_WIDTH_IN_FLOATS);
                xyi03 = __riscv_vfnmsac_vv_f32m8(xyi03, sample0xr, sample3yi, VECTOR_WIDTH_IN_FLOATS);
                yxr03 = __riscv_vfmacc_vv_f32m8(yxr03, sample0yi, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                yxi03 = __riscv_vfnmsac_vv_f32m8(yxi03, sample0yr, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                yyr03 = __riscv_vfmacc_vv_f32m8(yyr03, sample0yi, sample3yi, VECTOR_WIDTH_IN_FLOATS);
                yyi03 = __riscv_vfnmsac_vv_f32m8(yyi03, sample0yr, sample3yi, VECTOR_WIDTH_IN_FLOATS);

                xxr13 = __riscv_vfmacc_vv_f32m8(xxr13, sample1xr, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                xxi13 = __riscv_vfmacc_vv_f32m8(xxi13, sample1xi, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                xyr13 = __riscv_vfmacc_vv_f32m8(xyr13, sample1xr, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                xyi13 = __riscv_vfmacc_vv_f32m8(xyi13, sample1xi, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                yxr13 = __riscv_vfmacc_vv_f32m8(yxr13, sample1yr, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                yxi13 = __riscv_vfmacc_vv_f32m8(yxi13, sample1yi, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                yyr13 = __riscv_vfmacc_vv_f32m8(yyr13, sample1yr, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                yyi13 = __riscv_vfmacc_vv_f32m8(yyi13, sample1yi, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                xxr13 = __riscv_vfmacc_vv_f32m8(xxr13, sample1xi, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                xxi13 = __riscv_vfnmsac_vv_f32m8(xxi13, sample1xr, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                xyr13 = __riscv_vfmacc_vv_f32m8(xyr13, sample1xi, sample3yi, VECTOR_WIDTH_IN_FLOATS);
                xyi13 = __riscv_vfnmsac_vv_f32m8(xyi13, sample1xr, sample3yi, VECTOR_WIDTH_IN_FLOATS);
                yxr13 = __riscv_vfmacc_vv_f32m8(yxr13, sample1yi, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                yxi13 = __riscv_vfnmsac_vv_f32m8(yxi13, sample1yr, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                yyr13 = __riscv_vfmacc_vv_f32m8(yyr13, sample1yi, sample3yi, VECTOR_WIDTH_IN_FLOATS);
                yyi13 = __riscv_vfnmsac_vv_f32m8(yyi13, sample1yr, sample3yi, VECTOR_WIDTH_IN_FLOATS);

                xxr23 = __riscv_vfmacc_vv_f32m8(xxr23, sample2xr, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                xxi23 = __riscv_vfmacc_vv_f32m8(xxi23, sample2xi, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                xyr23 = __riscv_vfmacc_vv_f32m8(xyr23, sample2xr, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                xyi23 = __riscv_vfmacc_vv_f32m8(xyi23, sample2xi, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                yxr23 = __riscv_vfmacc_vv_f32m8(yxr23, sample2yr, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                yxi23 = __riscv_vfmacc_vv_f32m8(yxi23, sample2yi, sample3xr, VECTOR_WIDTH_IN_FLOATS);
                yyr23 = __riscv_vfmacc_vv_f32m8(yyr23, sample2yr, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                yyi23 = __riscv_vfmacc_vv_f32m8(yyi23, sample2yi, sample3yr, VECTOR_WIDTH_IN_FLOATS);
                xxr23 = __riscv_vfmacc_vv_f32m8(xxr23, sample2xi, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                xxi23 = __riscv_vfnmsac_vv_f32m8(xxi23, sample2xr, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                xyr23 = __riscv_vfmacc_vv_f32m8(xyr23, sample2xi, sample3yi, VECTOR_WIDTH_IN_FLOATS);
                xyi23 = __riscv_vfnmsac_vv_f32m8(xyi23, sample2xr, sample3yi, VECTOR_WIDTH_IN_FLOATS);
                yxr23 = __riscv_vfmacc_vv_f32m8(yxr23, sample2yi, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                yxi23 = __riscv_vfnmsac_vv_f32m8(yxi23, sample2yr, sample3xi, VECTOR_WIDTH_IN_FLOATS);
                yyr23 = __riscv_vfmacc_vv_f32m8(yyr23, sample2yi, sample3yi, VECTOR_WIDTH_IN_FLOATS);
                yyi23 = __riscv_vfnmsac_vv_f32m8(yyi23, sample2yr, sample3yi, VECTOR_WIDTH_IN_FLOATS);


                xxr04 = __riscv_vfmacc_vv_f32m8(xxr04, sample0xr, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                xxi04 = __riscv_vfmacc_vv_f32m8(xxi04, sample0xi, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                xyr04 = __riscv_vfmacc_vv_f32m8(xyr04, sample0xr, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                xyi04 = __riscv_vfmacc_vv_f32m8(xyi04, sample0xi, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                yxr04 = __riscv_vfmacc_vv_f32m8(yxr04, sample0yr, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                yxi04 = __riscv_vfmacc_vv_f32m8(yxi04, sample0yi, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                yyr04 = __riscv_vfmacc_vv_f32m8(yyr04, sample0yr, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                yyi04 = __riscv_vfmacc_vv_f32m8(yyi04, sample0yi, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                xxr04 = __riscv_vfmacc_vv_f32m8(xxr04, sample0xi, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                xxi04 = __riscv_vfnmsac_vv_f32m8(xxi04, sample0xr, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                xyr04 = __riscv_vfmacc_vv_f32m8(xyr04, sample0xi, sample4yi, VECTOR_WIDTH_IN_FLOATS);
                xyi04 = __riscv_vfnmsac_vv_f32m8(xyi04, sample0xr, sample4yi, VECTOR_WIDTH_IN_FLOATS);
                yxr04 = __riscv_vfmacc_vv_f32m8(yxr04, sample0yi, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                yxi04 = __riscv_vfnmsac_vv_f32m8(yxi04, sample0yr, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                yyr04 = __riscv_vfmacc_vv_f32m8(yyr04, sample0yi, sample4yi, VECTOR_WIDTH_IN_FLOATS);
                yyi04 = __riscv_vfnmsac_vv_f32m8(yyi04, sample0yr, sample4yi, VECTOR_WIDTH_IN_FLOATS);

                xxr14 = __riscv_vfmacc_vv_f32m8(xxr14, sample1xr, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                xxi14 = __riscv_vfmacc_vv_f32m8(xxi14, sample1xi, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                xyr14 = __riscv_vfmacc_vv_f32m8(xyr14, sample1xr, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                xyi14 = __riscv_vfmacc_vv_f32m8(xyi14, sample1xi, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                yxr14 = __riscv_vfmacc_vv_f32m8(yxr14, sample1yr, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                yxi14 = __riscv_vfmacc_vv_f32m8(yxi14, sample1yi, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                yyr14 = __riscv_vfmacc_vv_f32m8(yyr14, sample1yr, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                yyi14 = __riscv_vfmacc_vv_f32m8(yyi14, sample1yi, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                xxr14 = __riscv_vfmacc_vv_f32m8(xxr14, sample1xi, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                xxi14 = __riscv_vfnmsac_vv_f32m8(xxi14, sample1xr, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                xyr14 = __riscv_vfmacc_vv_f32m8(xyr14, sample1xi, sample4yi, VECTOR_WIDTH_IN_FLOATS);
                xyi14 = __riscv_vfnmsac_vv_f32m8(xyi14, sample1xr, sample4yi, VECTOR_WIDTH_IN_FLOATS);
                yxr14 = __riscv_vfmacc_vv_f32m8(yxr14, sample1yi, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                yxi14 = __riscv_vfnmsac_vv_f32m8(yxi14, sample1yr, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                yyr14 = __riscv_vfmacc_vv_f32m8(yyr14, sample1yi, sample4yi, VECTOR_WIDTH_IN_FLOATS);
                yyi14 = __riscv_vfnmsac_vv_f32m8(yyi14, sample1yr, sample4yi, VECTOR_WIDTH_IN_FLOATS);

                xxr24 = __riscv_vfmacc_vv_f32m8(xxr24, sample2xr, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                xxi24 = __riscv_vfmacc_vv_f32m8(xxi24, sample2xi, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                xyr24 = __riscv_vfmacc_vv_f32m8(xyr24, sample2xr, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                xyi24 = __riscv_vfmacc_vv_f32m8(xyi24, sample2xi, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                yxr24 = __riscv_vfmacc_vv_f32m8(yxr24, sample2yr, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                yxi24 = __riscv_vfmacc_vv_f32m8(yxi24, sample2yi, sample4xr, VECTOR_WIDTH_IN_FLOATS);
                yyr24 = __riscv_vfmacc_vv_f32m8(yyr24, sample2yr, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                yyi24 = __riscv_vfmacc_vv_f32m8(yyi24, sample2yi, sample4yr, VECTOR_WIDTH_IN_FLOATS);
                xxr24 = __riscv_vfmacc_vv_f32m8(xxr24, sample2xi, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                xxi24 = __riscv_vfnmsac_vv_f32m8(xxi24, sample2xr, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                xyr24 = __riscv_vfmacc_vv_f32m8(xyr24, sample2xi, sample4yi, VECTOR_WIDTH_IN_FLOATS);
                xyi24 = __riscv_vfnmsac_vv_f32m8(xyi24, sample2xr, sample4yi, VECTOR_WIDTH_IN_FLOATS);
                yxr24 = __riscv_vfmacc_vv_f32m8(yxr24, sample2yi, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                yxi24 = __riscv_vfnmsac_vv_f32m8(yxi24, sample2yr, sample4xi, VECTOR_WIDTH_IN_FLOATS);
                yyr24 = __riscv_vfmacc_vv_f32m8(yyr24, sample2yi, sample4yi, VECTOR_WIDTH_IN_FLOATS);
                yyi24 = __riscv_vfnmsac_vv_f32m8(yyi24, sample2yr, sample4yi, VECTOR_WIDTH_IN_FLOATS);
	    }

	    if(cell < nrCells) {
		missedBaselines[BASELINE(stat0+0, stat3+0)] = false;
		missedBaselines[BASELINE(stat0+1, stat3+0)] = false;
		missedBaselines[BASELINE(stat0+2, stat3+0)] = false;
		missedBaselines[BASELINE(stat0+0, stat3+1)] = false;
		missedBaselines[BASELINE(stat0+1, stat3+1)] = false;
		missedBaselines[BASELINE(stat0+2, stat3+1)] = false;

		size_t visIndex;

		// now, we have to sum the (vector width) values inside the regs.
		// __m128 tmp0, tmp1, xx_xy, tmp2, tmp3, yx_yy;

		visIndex = VISIBILITIES_INDEX(BASELINE(stat0, stat3), channel, 0, 0, 0);
		visibilities[visIndex+0] = reduce_vfloat32m8(xxr03);
		visibilities[visIndex+1] = reduce_vfloat32m8(xxi03);
		visibilities[visIndex+2] = reduce_vfloat32m8(xyr03);
		visibilities[visIndex+3] = reduce_vfloat32m8(xyi03);
		visibilities[visIndex+4] = reduce_vfloat32m8(yxr03);
		visibilities[visIndex+5] = reduce_vfloat32m8(yxi03);
		visibilities[visIndex+6] = reduce_vfloat32m8(yyr03);
		visibilities[visIndex+7] = reduce_vfloat32m8(yyi03);

		
		visIndex = VISIBILITIES_INDEX(BASELINE(stat0+1, stat3), channel, 0, 0, 0);
		visibilities[visIndex+0] = reduce_vfloat32m8(xxr13);
		visibilities[visIndex+1] = reduce_vfloat32m8(xxi13);
		visibilities[visIndex+2] = reduce_vfloat32m8(xyr13);
		visibilities[visIndex+3] = reduce_vfloat32m8(xyi13);
		visibilities[visIndex+4] = reduce_vfloat32m8(yxr13);
		visibilities[visIndex+5] = reduce_vfloat32m8(yxi13);
		visibilities[visIndex+6] = reduce_vfloat32m8(yyr13);
		visibilities[visIndex+7] = reduce_vfloat32m8(yyi13);

		
		visIndex = VISIBILITIES_INDEX(BASELINE(stat0+2, stat3), channel, 0, 0, 0);
		visibilities[visIndex+0] = reduce_vfloat32m8(xxr23);
		visibilities[visIndex+1] = reduce_vfloat32m8(xxi23);
		visibilities[visIndex+2] = reduce_vfloat32m8(xyr23);
		visibilities[visIndex+3] = reduce_vfloat32m8(xyi23);
		visibilities[visIndex+4] = reduce_vfloat32m8(yxr23);
		visibilities[visIndex+5] = reduce_vfloat32m8(yxi23);
		visibilities[visIndex+6] = reduce_vfloat32m8(yyr23);
		visibilities[visIndex+7] = reduce_vfloat32m8(yyi23);
		
		
		visIndex = VISIBILITIES_INDEX(BASELINE(stat0, stat3+1), channel, 0, 0, 0);
		visibilities[visIndex+0] = reduce_vfloat32m8(xxr04);
		visibilities[visIndex+1] = reduce_vfloat32m8(xxi04);
		visibilities[visIndex+2] = reduce_vfloat32m8(xyr04);
		visibilities[visIndex+3] = reduce_vfloat32m8(xyi04);
		visibilities[visIndex+4] = reduce_vfloat32m8(yxr04);
		visibilities[visIndex+5] = reduce_vfloat32m8(yxi04);
		visibilities[visIndex+6] = reduce_vfloat32m8(yyr04);
		visibilities[visIndex+7] = reduce_vfloat32m8(yyi04);
		

		visIndex = VISIBILITIES_INDEX(BASELINE(stat0+1, stat3+1), channel, 0, 0, 0);
		visibilities[visIndex+0] = reduce_vfloat32m8(xxr14);
		visibilities[visIndex+1] = reduce_vfloat32m8(xxi14);
		visibilities[visIndex+2] = reduce_vfloat32m8(xyr14);
		visibilities[visIndex+3] = reduce_vfloat32m8(xyi14);
		visibilities[visIndex+4] = reduce_vfloat32m8(yxr14);
		visibilities[visIndex+5] = reduce_vfloat32m8(yxi14);
		visibilities[visIndex+6] = reduce_vfloat32m8(yyr14);
		visibilities[visIndex+7] = reduce_vfloat32m8(yyi14);
		
	
		visIndex = VISIBILITIES_INDEX(BASELINE(stat0+2, stat3+1), channel, 0, 0, 0);
		visibilities[visIndex+0] = reduce_vfloat32m8(xxr24);
		visibilities[visIndex+1] = reduce_vfloat32m8(xxi24);
		visibilities[visIndex+2] = reduce_vfloat32m8(xyr24);
		visibilities[visIndex+3] = reduce_vfloat32m8(xyi24);
		visibilities[visIndex+4] = reduce_vfloat32m8(yxr24);
		visibilities[visIndex+5] = reduce_vfloat32m8(yxi24);
		visibilities[visIndex+6] = reduce_vfloat32m8(yyr24);
		visibilities[visIndex+7] = reduce_vfloat32m8(yyi24);
	    }
	}
    }

    unsigned long long missedBytesLoaded, missedBytesStored;
    unsigned long long missedOps = computeMissedBaselines(samples, visibilities, missedBaselines,
							  nrTimes, nrTimesWidth, nrStations, nrChannels,
							  &missedBytesLoaded, &missedBytesStored);

    unsigned long long ops = missedOps + calcNrOps(nrTimes, nrStations, nrChannels, bytesLoaded, bytesStored);
    *bytesLoaded += missedBytesLoaded;
    *bytesLoaded += missedBytesStored;
   return ops;
}
