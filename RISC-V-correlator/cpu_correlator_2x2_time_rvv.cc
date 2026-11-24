#include <cstdlib>
#include <iostream>
#include <riscv_vector.h> // rvv

#include "riscv_correlator.h"

using namespace std;

static unsigned char cellToStatX[MAX_NR_CELLS], cellToStatY[MAX_NR_CELLS];

static unsigned fillCellToStatTable(unsigned nrStations)
{
    unsigned nrCells, stat0, stat2;

    for (stat2 = nrStations % 2 ? 1 : 2, nrCells = 0; stat2 < nrStations; stat2 += 2) {
	for (stat0 = 0; stat0 + 2 <= stat2; stat0 += 2, nrCells ++) {
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
	for (stat0 = 0; stat0 + 2 <= stat2; stat0 += 2, nrCells ++);
    }

    unsigned long long ops = (unsigned long long)nrChannels * nrCells * nrTimes * 32L * 4L;

    *bytesLoaded = (unsigned long long)nrChannels * nrCells * nrTimes * 4L * 4L * 4L;
    *bytesStored = (unsigned long long)nrChannels * nrCells * 2L * 4L * 4L;

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

unsigned long long cpuCorrelator_2x2_time_rvv(const float* __restrict__ samples, float* __restrict__ visibilities, 
						      const unsigned nrTimes, const unsigned nrTimesWidth, 
						      const unsigned nrStations, const unsigned nrChannels,
						      unsigned long long* bytesLoaded, unsigned long long* bytesStored)
{
    const unsigned nrBaselines = NR_BASELINES(nrStations);
    bool missedBaselines[nrBaselines*2]; // we sometimes go beyond the #baslines, but then we don't store the visibilities, so that is ok.
    for(unsigned baseline=0; baseline < nrBaselines; baseline++) {
	missedBaselines[baseline] = true;
    }

    const unsigned nrCells = fillCellToStatTable(nrStations);

    for (unsigned channel = 0; channel < nrChannels; channel++) {
	for (unsigned cell = 0; cell < nrCells; cell++) {
	    unsigned stat0 = cellToStatX[cell];
	    unsigned stat2 = cellToStatY[cell];

	    size_t index0 = SAMPLE_INDEX(stat0+0, channel, 0, 0, 0);
	    size_t index1 = SAMPLE_INDEX(stat0+1, channel, 0, 0, 0);
	    size_t index2 = SAMPLE_INDEX(stat2+0, channel, 0, 0, 0);
	    size_t index3 = SAMPLE_INDEX(stat2+1, channel, 0, 0, 0);

	    vfloat32m8_t xxr02 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xxi02 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyr02 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyi02 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxr02 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxi02 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyr02 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyi02 = i_vfloat32m8(0.0f);

	    vfloat32m8_t xxr12 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xxi12 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyr12 = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyi12 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxr12 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxi12 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyr12 = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyi12 = i_vfloat32m8(0.0f);

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

	    // assume nrTimes is divisable by vector width
	    for (unsigned time = 0; time < nrTimes; time += VECTOR_WIDTH_IN_FLOATS) {
	      // This assumes a different memory layout with (vector width) samples in time behind each other:
	      // (xr1 xr2 xr3 xr4) (xi1 xi2 xi3 xi4) (yr1 yr2 yr3 yr4) (yi1 yi2 yi3 yi4), etc.
		vfloat32m8_t sample0xr = __riscv_vle32_v_f32m8(samples + time*4 + index0+0, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample0xi = __riscv_vle32_v_f32m8(samples + time*4 + index0+8, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample0yr = __riscv_vle32_v_f32m8(samples + time*4 + index0+16, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample0yi = __riscv_vle32_v_f32m8(samples + time*4 + index0+24, VECTOR_WIDTH_IN_FLOATS);

		vfloat32m8_t sample1xr = __riscv_vle32_v_f32m8(samples + time*4 + index1+0, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample1xi = __riscv_vle32_v_f32m8(samples + time*4 + index1+8, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample1yr = __riscv_vle32_v_f32m8(samples + time*4 + index1+16, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample1yi = __riscv_vle32_v_f32m8(samples + time*4 + index1+24, VECTOR_WIDTH_IN_FLOATS);

		vfloat32m8_t sample2xr = __riscv_vle32_v_f32m8(samples + time*4 + index2+0, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample2xi = __riscv_vle32_v_f32m8(samples + time*4 + index2+8, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample2yr = __riscv_vle32_v_f32m8(samples + time*4 + index2+16, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample2yi = __riscv_vle32_v_f32m8(samples + time*4 + index2+24, VECTOR_WIDTH_IN_FLOATS);

		vfloat32m8_t sample3xr = __riscv_vle32_v_f32m8(samples + time*4 + index3+0, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample3xi = __riscv_vle32_v_f32m8(samples + time*4 + index3+8, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample3yr = __riscv_vle32_v_f32m8(samples + time*4 + index3+16, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample3yi = __riscv_vle32_v_f32m8(samples + time*4 + index3+24, VECTOR_WIDTH_IN_FLOATS);

                xxr02 = __riscv_vfmacc_vv_f32m8(xxr02, sample0xr, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                xxi02 = __riscv_vfmacc_vv_f32m8(xxi02, sample0xi, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                xyr02 = __riscv_vfmacc_vv_f32m8(xyr02, sample0xr, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                xyi02 = __riscv_vfmacc_vv_f32m8(xyi02, sample0xi, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                yxr02 = __riscv_vfmacc_vv_f32m8(yxr02, sample0yr, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                yxi02 = __riscv_vfmacc_vv_f32m8(yxi02, sample0yi, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                yyr02 = __riscv_vfmacc_vv_f32m8(yyr02, sample0yr, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                yyi02 = __riscv_vfmacc_vv_f32m8(yyi02, sample0yi, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                xxr02 = __riscv_vfmacc_vv_f32m8(xxr02, sample0xi, sample2xi, VECTOR_WIDTH_IN_FLOATS);
                xxi02 = __riscv_vfnmsac_vv_f32m8(xxi02, sample0xr, sample2xi, VECTOR_WIDTH_IN_FLOATS);
                xyr02 = __riscv_vfmacc_vv_f32m8(xyr02, sample0xi, sample2yi, VECTOR_WIDTH_IN_FLOATS);
                xyi02 = __riscv_vfnmsac_vv_f32m8(xyi02, sample0xr, sample2yi, VECTOR_WIDTH_IN_FLOATS);
                yxr02 = __riscv_vfmacc_vv_f32m8(yxr02, sample0yi, sample2xi, VECTOR_WIDTH_IN_FLOATS);
                yxi02 = __riscv_vfnmsac_vv_f32m8(yxi02, sample0yr, sample2xi, VECTOR_WIDTH_IN_FLOATS);
                yyr02 = __riscv_vfmacc_vv_f32m8(yyr02, sample0yi, sample2yi, VECTOR_WIDTH_IN_FLOATS);
                yyi02 = __riscv_vfnmsac_vv_f32m8(yyi02, sample0yr, sample2yi, VECTOR_WIDTH_IN_FLOATS);

                xxr12 = __riscv_vfmacc_vv_f32m8(xxr12, sample1xr, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                xxi12 = __riscv_vfmacc_vv_f32m8(xxi12, sample1xi, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                xyr12 = __riscv_vfmacc_vv_f32m8(xyr12, sample1xr, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                xyi12 = __riscv_vfmacc_vv_f32m8(xyi12, sample1xi, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                yxr12 = __riscv_vfmacc_vv_f32m8(yxr12, sample1yr, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                yxi12 = __riscv_vfmacc_vv_f32m8(yxi12, sample1yi, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                yyr12 = __riscv_vfmacc_vv_f32m8(yyr12, sample1yr, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                yyi12 = __riscv_vfmacc_vv_f32m8(yyi12, sample1yi, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                xxr12 = __riscv_vfmacc_vv_f32m8(xxr12, sample1xi, sample2xi, VECTOR_WIDTH_IN_FLOATS);
                xxi12 = __riscv_vfnmsac_vv_f32m8(xxi12, sample1xr, sample2xi, VECTOR_WIDTH_IN_FLOATS);
                xyr12 = __riscv_vfmacc_vv_f32m8(xyr12, sample1xi, sample2yi, VECTOR_WIDTH_IN_FLOATS);
                xyi12 = __riscv_vfnmsac_vv_f32m8(xyi12, sample1xr, sample2yi, VECTOR_WIDTH_IN_FLOATS);
                yxr12 = __riscv_vfmacc_vv_f32m8(yxr12, sample1yi, sample2xi, VECTOR_WIDTH_IN_FLOATS);
                yxi12 = __riscv_vfnmsac_vv_f32m8(yxi12, sample1yr, sample2xi, VECTOR_WIDTH_IN_FLOATS);
                yyr12 = __riscv_vfmacc_vv_f32m8(yyr12, sample1yi, sample2yi, VECTOR_WIDTH_IN_FLOATS);
                yyi12 = __riscv_vfnmsac_vv_f32m8(yyi12, sample1yr, sample2yi, VECTOR_WIDTH_IN_FLOATS);

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
	    }

	    if (cell < nrCells) {
		missedBaselines[BASELINE(stat0, stat2)] = false;
		missedBaselines[BASELINE(stat0+1, stat2)] = false;
		missedBaselines[BASELINE(stat0, stat2+1)] = false;
		missedBaselines[BASELINE(stat0+1, stat2+1)] = false;

		// now, we have to sum the (vector width) values inside the regs.
		//vfloat32m8_t tmp0, tmp1, xx_xy, tmp2, tmp3, yx_yy;
		size_t visIndex;

		visIndex = VISIBILITIES_INDEX(BASELINE(stat0, stat2), channel, 0, 0, 0);
		/* 
		tmp0  = _mm_hadd_ps(xxr02, xxi02);   // xxr0+xxr1, xxr2+xxr3, xxi0+xxi1, xxi2+xxi3
		tmp1  = _mm_hadd_ps(xyr02, xyi02);   // xyr0+xyr1, xyr2+xyr3, xyi0+xyi1, xyi2+xyi3
		xx_xy = _mm_hadd_ps(tmp0, tmp1);     // xxr, xxi, xyr,xyi
		_mm_store_ps(&(visibilities[visIndex+0]), xx_xy);

		tmp2  = _mm_hadd_ps(yxr02, yxi02);
		tmp3  = _mm_hadd_ps(yyr02, yyi02);
		yx_yy = _mm_hadd_ps(tmp2, tmp3);
		_mm_store_ps(&(visibilities[visIndex+4]), yx_yy);
		*/
		//cout << "visIndex stat0,stat2: xx_xy:\t" << visIndex << ", yx_yy: " << (visIndex+4) << endl;
		visibilities[visIndex+0] = reduce_vfloat32m8(xxr02);
		visibilities[visIndex+1] = reduce_vfloat32m8(xxi02);
		visibilities[visIndex+2] = reduce_vfloat32m8(xyr02);
		visibilities[visIndex+3] = reduce_vfloat32m8(xyi02);
		visibilities[visIndex+4] = reduce_vfloat32m8(yxr02);
		visibilities[visIndex+5] = reduce_vfloat32m8(yxi02);
		visibilities[visIndex+6] = reduce_vfloat32m8(yyr02);
		visibilities[visIndex+7] = reduce_vfloat32m8(yyi02);
		
		visIndex = VISIBILITIES_INDEX(BASELINE(stat0+1, stat2), channel, 0, 0, 0);
		/*
		tmp0  = _mm_hadd_ps(xxr12, xxi12);   // xxr0+xxr1, xxr2+xxr3, xxi0+xxi1, xxi2+xxi3
		tmp1  = _mm_hadd_ps(xyr12, xyi12);   // xyr0+xyr1, xyr2+xyr3, xyi0+xyi1, xyi2+xyi3
		xx_xy = _mm_hadd_ps(tmp0, tmp1);     // xxr, xxi, xyr,xyi
		_mm_store_ps(&(visibilities[visIndex+0]), xx_xy);

		tmp2  = _mm_hadd_ps(yxr12, yxi12);
		tmp3  = _mm_hadd_ps(yyr12, yyi12);
		yx_yy = _mm_hadd_ps(tmp2, tmp3);
		_mm_store_ps(&(visibilities[visIndex+4]), yx_yy);
		*/
		//cout << "visIndex stat0+1,stat2: xx_xy:\t" << visIndex << ", yx_yy: " << (visIndex+4) << endl;
		visibilities[visIndex+0] = reduce_vfloat32m8(xxr12);
		visibilities[visIndex+1] = reduce_vfloat32m8(xxi12);
		visibilities[visIndex+2] = reduce_vfloat32m8(xyr12);
		visibilities[visIndex+3] = reduce_vfloat32m8(xyi12);
		visibilities[visIndex+4] = reduce_vfloat32m8(yxr12);
		visibilities[visIndex+5] = reduce_vfloat32m8(yxi12);
		visibilities[visIndex+6] = reduce_vfloat32m8(yyr12);
		visibilities[visIndex+7] = reduce_vfloat32m8(yyi12);
		
		visIndex = VISIBILITIES_INDEX(BASELINE(stat0, stat2+1), channel, 0, 0, 0);
		/*
		tmp0  = _mm_hadd_ps(xxr03, xxi03);   // xxr0+xxr1, xxr2+xxr3, xxi0+xxi1, xxi2+xxi3
		tmp1  = _mm_hadd_ps(xyr03, xyi03);   // xyr0+xyr1, xyr2+xyr3, xyi0+xyi1, xyi2+xyi3
		xx_xy = _mm_hadd_ps(tmp0, tmp1);     // xxr, xxi, xyr,xyi
		_mm_store_ps(&(visibilities[visIndex+0]), xx_xy);
		
		tmp2  = _mm_hadd_ps(yxr03, yxi03);
		tmp3  = _mm_hadd_ps(yyr03, yyi03);
		yx_yy = _mm_hadd_ps(tmp2, tmp3);
		_mm_store_ps(&(visibilities[visIndex+4]), yx_yy);
		*/
		//cout << "visIndex stat0,stat2+1: xx_xy:\t" << visIndex << ", yx_yy: " << (visIndex+4) << endl;
		visibilities[visIndex+0] = reduce_vfloat32m8(xxr03);
		visibilities[visIndex+1] = reduce_vfloat32m8(xxi03);
		visibilities[visIndex+2] = reduce_vfloat32m8(xyr03);
		visibilities[visIndex+3] = reduce_vfloat32m8(xyi03);
		visibilities[visIndex+4] = reduce_vfloat32m8(yxr03);
		visibilities[visIndex+5] = reduce_vfloat32m8(yxi03);
		visibilities[visIndex+6] = reduce_vfloat32m8(yyr03);
		visibilities[visIndex+7] = reduce_vfloat32m8(yyi03);
		
		visIndex = VISIBILITIES_INDEX(BASELINE(stat0+1, stat2+1), channel, 0, 0, 0);
		/*
		tmp0  = _mm_hadd_ps(xxr13, xxi13);   // xxr0+xxr1, xxr2+xxr3, xxi0+xxi1, xxi2+xxi3
		tmp1  = _mm_hadd_ps(xyr13, xyi13);   // xyr0+xyr1, xyr2+xyr3, xyi0+xyi1, xyi2+xyi3
		xx_xy = _mm_hadd_ps(tmp0, tmp1);     // xxr, xxi, xyr,xyi
		_mm_store_ps(&(visibilities[visIndex+0]), xx_xy);
		
		tmp2  = _mm_hadd_ps(yxr13, yxi13);
		tmp3  = _mm_hadd_ps(yyr13, yyi13);
		yx_yy = _mm_hadd_ps(tmp2, tmp3);
		_mm_store_ps(&(visibilities[visIndex+4]), yx_yy);
		*/
		//cout << "visIndex stat0+1,stat2+1: xx_xy:\t" << visIndex << ", yx_yy: " << (visIndex+4) << endl;
		visibilities[visIndex+0] = reduce_vfloat32m8(xxr13);
		visibilities[visIndex+1] = reduce_vfloat32m8(xxi13);
		visibilities[visIndex+2] = reduce_vfloat32m8(xyr13);
		visibilities[visIndex+3] = reduce_vfloat32m8(xyi13);
		visibilities[visIndex+4] = reduce_vfloat32m8(yxr13);
		visibilities[visIndex+5] = reduce_vfloat32m8(yxi13);
		visibilities[visIndex+6] = reduce_vfloat32m8(yyr13);
		visibilities[visIndex+7] = reduce_vfloat32m8(yyi13);
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
