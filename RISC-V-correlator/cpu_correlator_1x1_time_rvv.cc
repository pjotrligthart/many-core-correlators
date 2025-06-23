#include <cstdlib>
#include <iostream>
#include <riscv_vector.h> // rvv

#include "riscv_correlator.h"


using namespace std;

// used for 1x1 correlator only
static unsigned char baselineToStat1[MAX_NR_CELLS], baselineToStat2[MAX_NR_CELLS];

static unsigned fillCellToStatTable(const unsigned nrStations)
{
    unsigned baseline = 0;

    for (unsigned stat2 = 0; stat2 < nrStations; stat2 ++) {
	for (unsigned stat1 = 0; stat1 <= stat2; stat1 ++, baseline ++) {
	    baselineToStat1[baseline] = stat1;
	    baselineToStat2[baseline] = stat2;
	}
    }

    return baseline;
}

vfloat32m8_t i_vfloat32m8(const float val)
{
    float result[VECTOR_WIDTH_IN_FLOATS];
    for(int i = 0; i < VECTOR_WIDTH_IN_FLOATS; i++) {
	result[i] = val;
    }

    return __riscv_vle32_v_f32m8(result, VECTOR_WIDTH_IN_FLOATS);
}

// (Temporary?) function that reduces/sums all values in a vfloat32m8 to a single float
float reduce_vfloat32m8(const vfloat32m8_t &src)
{
    vfloat32m1_t zero = __riscv_vfmv_s_f_f32m1(0.0f, 1);
    vfloat32m1_t res = __riscv_vfmv_s_f_f32m1(0.0f, 1);

    // Use vfredusum because it seems to be faster on our target board (BPI-F3)
    // See benchmarks at https://camel-cdr.github.io/rvv-bench-results/bpi_f3/index.html
    res = __riscv_vfredusum_vs_f32m8_f32m1(src, zero, 8);
    return __riscv_vfmv_f_s_f32m1_f32(res);
}


// assume samples are in the correct order: xr0, xr1, xr2, xr3; xi0, xi1, xi2, xi3; yr0, yr1, yr2, yr3; ui0, yi1, yi2, yi3

unsigned long long cpuCorrelator_1x1_time_rvv(const float* __restrict__ samples, float* __restrict__ visibilities,
						      const unsigned nrTimes, const unsigned nrTimesWidth,
						      const unsigned nrStations, const unsigned nrChannels,
						      unsigned long long* bytesLoaded, unsigned long long* bytesStored)
{
    unsigned nrBaselines = fillCellToStatTable(nrStations);

    for (unsigned channel = 0; channel < nrChannels; channel ++) {
	for (unsigned baseline = 0; baseline < nrBaselines; baseline++) {
	    unsigned stat1 = baselineToStat1[baseline];
	    unsigned stat2 = baselineToStat2[baseline];

	    size_t index1 = SAMPLE_INDEX(stat1, channel, 0, 0, 0);
	    size_t index2 = SAMPLE_INDEX(stat2, channel, 0, 0, 0);

	    vfloat32m8_t xxr = i_vfloat32m8(0.0f);
	    vfloat32m8_t xxi = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyr = i_vfloat32m8(0.0f);
	    vfloat32m8_t xyi = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxr = i_vfloat32m8(0.0f);
	    vfloat32m8_t yxi = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyr = i_vfloat32m8(0.0f);
	    vfloat32m8_t yyi = i_vfloat32m8(0.0f);

	    // assume nrTimes is divisable by vector width
	    for (unsigned time = 0; time < nrTimes; time += VECTOR_WIDTH_IN_FLOATS) {
	      // This assumes a different memory layout with (vector width) samples in time behind each other:
	      // (xr1 xr2 xr3 xr4) (xi1 xi2 xi3 xi4) (yr1 yr2 yr3 yr4) (yi1 yi2 yi3 yi4), etc.
		vfloat32m8_t sample1xr = __riscv_vle32_v_f32m8(samples + time*4 + index1+0, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample1xi = __riscv_vle32_v_f32m8(samples + time*4 + index1+8, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample1yr = __riscv_vle32_v_f32m8(samples + time*4 + index1+16, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample1yi = __riscv_vle32_v_f32m8(samples + time*4 + index1+24, VECTOR_WIDTH_IN_FLOATS);

		vfloat32m8_t sample2xr = __riscv_vle32_v_f32m8(samples + time*4 + index2+0, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample2xi = __riscv_vle32_v_f32m8(samples + time*4 + index2+8, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample2yr = __riscv_vle32_v_f32m8(samples + time*4 + index2+16, VECTOR_WIDTH_IN_FLOATS);
		vfloat32m8_t sample2yi = __riscv_vle32_v_f32m8(samples + time*4 + index2+24, VECTOR_WIDTH_IN_FLOATS);

#if 1
                xxr = __riscv_vfmacc_vv_f32m8(xxr, sample1xr, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                xxi = __riscv_vfmacc_vv_f32m8(xxi, sample1xi, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                xyr = __riscv_vfmacc_vv_f32m8(xyr, sample1xr, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                xyi = __riscv_vfmacc_vv_f32m8(xyi, sample1xi, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                yxr = __riscv_vfmacc_vv_f32m8(yxr, sample1yr, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                yxi = __riscv_vfmacc_vv_f32m8(yxi, sample1yi, sample2xr, VECTOR_WIDTH_IN_FLOATS);
                yyr = __riscv_vfmacc_vv_f32m8(yyr, sample1yr, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                yyi = __riscv_vfmacc_vv_f32m8(yyi, sample1yi, sample2yr, VECTOR_WIDTH_IN_FLOATS);
                xxr = __riscv_vfmacc_vv_f32m8(xxr, sample1xi, sample2xi, VECTOR_WIDTH_IN_FLOATS);

                xxi = __riscv_vfnmsac_vv_f32m8(xxi, sample1xr, sample2xi, VECTOR_WIDTH_IN_FLOATS);

                xyr = __riscv_vfmacc_vv_f32m8(xyr, sample1xi, sample2yi, VECTOR_WIDTH_IN_FLOATS);

                xyi = __riscv_vfnmsac_vv_f32m8(xyi, sample1xr, sample2yi, VECTOR_WIDTH_IN_FLOATS);

                yxr = __riscv_vfmacc_vv_f32m8(yxr, sample1yi, sample2xi, VECTOR_WIDTH_IN_FLOATS);

                yxi = __riscv_vfnmsac_vv_f32m8(yxi, sample1yr, sample2xi, VECTOR_WIDTH_IN_FLOATS);

                yyr = __riscv_vfmacc_vv_f32m8(yyr, sample1yi, sample2yi, VECTOR_WIDTH_IN_FLOATS);

                yyi = __riscv_vfnmsac_vv_f32m8(yyi, sample1yr, sample2yi, VECTOR_WIDTH_IN_FLOATS);
#else
/*
		xxr = _mm_add_ps(xxr, _mm_add_ps(_mm_mul_ps(sample1xr, sample2xr), _mm_mul_ps(sample1xi, sample2xi)));
		xxi = _mm_add_ps(xxi, _mm_sub_ps(_mm_mul_ps(sample1xi, sample2xr), _mm_mul_ps(sample1xr, sample2xi)));

		xyr = _mm_add_ps(xyr, _mm_add_ps(_mm_mul_ps(sample1xr, sample2yr), _mm_mul_ps(sample1xi, sample2yi)));
		xyi = _mm_add_ps(xyi, _mm_sub_ps(_mm_mul_ps(sample1xi, sample2yr), _mm_mul_ps(sample1xr, sample2yi)));

		yxr = _mm_add_ps(yxr, _mm_add_ps(_mm_mul_ps(sample1yr, sample2xr), _mm_mul_ps(sample1yi, sample2xi)));
		yxi = _mm_add_ps(yxi, _mm_sub_ps(_mm_mul_ps(sample1yi, sample2xr), _mm_mul_ps(sample1yr, sample2xi)));

		yyr = _mm_add_ps(yyr, _mm_add_ps(_mm_mul_ps(sample1yr, sample2yr), _mm_mul_ps(sample1yi, sample2yi)));
		yyi = _mm_add_ps(yyi, _mm_sub_ps(_mm_mul_ps(sample1yi, sample2yr), _mm_mul_ps(sample1yr, sample2yi)));
*/
#endif
	    }

	    size_t vis_index = VISIBILITIES_INDEX(baseline, channel, 0, 0, 0);
	    
	    float res_xxr = reduce_vfloat32m8(xxr);
	    float res_xxi = reduce_vfloat32m8(xxi);
	    float res_xyr = reduce_vfloat32m8(xyr);
	    float res_xyi = reduce_vfloat32m8(xyi);
	    float res_yxr = reduce_vfloat32m8(yxr);
	    float res_yxi = reduce_vfloat32m8(yxi);
	    float res_yyr = reduce_vfloat32m8(yyr);
	    float res_yyi = reduce_vfloat32m8(yyi);
	    
	    visibilities[vis_index+0] = res_xxr;
	    visibilities[vis_index+1] = res_xxi;
	    visibilities[vis_index+2] = res_xyr;
	    visibilities[vis_index+3] = res_xyi;
	    visibilities[vis_index+4] = res_yxr;
	    visibilities[vis_index+5] = res_yxi;
	    visibilities[vis_index+6] = res_yyr;
	    visibilities[vis_index+7] = res_yyi;
	}
    }

    *bytesLoaded = nrChannels * nrBaselines * nrTimes * 8 * sizeof(float); // samples
    *bytesStored = nrChannels * nrBaselines * 8 * sizeof(float); // vis

    unsigned long long ops = nrChannels * nrBaselines * nrTimes * 16L * 2L;
    return ops;
}
