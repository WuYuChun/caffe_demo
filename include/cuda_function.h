#ifndef CUDA_FUCTION_H_
#define CUDA_FUCTION_H_

#include <stdint.h>

void ArgmaBilinearResizeGPU(const float *src, int batch, int src_height, int src_width, int channel,
                            uint8_t* dst, int dst_height, int dst_width);



#endif
