#include "cuda_function.h"

#include <cuda_runtime.h>

#include <cmath>


__global__ void ArgmaxBilinearResizeKernel(const float*src , int batch, int src_height, int src_width, int channel,
                                          uint8_t *dst, int dst_height, int dst_width){

    int idx = threadIdx.x + blockIdx.x*blockDim.x;
    int idy = threadIdx.y + blockIdx.y*blockDim.y;                                          

    if(idx < dst_width && idy < dst_height){
        float scale_x = 1.0f * src_width / dst_width;
        float scale_y = 1.0f * src_height / dst_height;

        int src_step = src_height * src_width * channel;
        int src_single_step = src_height * src_width;
        int dst_step = dst_height * dst_width;

        float src_idx = (idx+0.5f)*scale_x - 0.5f;
        float src_idy = (idy+0.5f)*scale_y - 0.5f;

        int ix = src_idx;
        int iy = src_idy;

        float fv = src_idx - ix;
        float fu = src_idy - iy;

        int inimg_id00 = (iy + 0)*src_width + ix +0;
        int inimg_id01 = (iy + 0)*src_width + ix +1;
        int inimg_id10 = (iy + 1)*src_width + ix +0;
        int inimg_id11 = (iy + 1)*src_width + ix +1;
        float scale00 = (1.0f -fu)*(1.0f - fv);
        float scale01 = (1.0f -fu)*(       fv);
        float scale10 = (      fu)*(1.0f - fv);
        float scale11 = (      fu)*(       fv);
        for(int b = 0; b < batch; ++b){
            float max_value = -9999.0f;
            uint8_t index = 0;
            for (uint8_t i = 0; i < channel; i++){
                float value = src[inimg_id00 + i * src_single_step + b * src_step] * scale00 +
                              src[inimg_id01 + i * src_single_step + b * src_step] * scale01 +
                              src[inimg_id10 + i * src_single_step + b * src_step] * scale10 +
                              src[inimg_id11 + i * src_single_step + b * src_step] * scale11 ;
                if(value > max_value){
                    max_value = value;
                    index = i;
                }              
            }
            dst[idx+idy*dst_width+b*dst_step] = index;
            
        }

    }
}

void ArgmaBilinearResizeGPU(const float *src, int batch, int src_height, int src_width, int channel,
                            uint8_t* dst, int dst_height, int dst_width){

    int block_width = 16;
    int block_height = 16;
    int grid_width = std::ceil(1.0f*dst_width /block_height);    
    int grid_height = std::ceil(1.0f*dst_height/block_height);

    dim3 block(block_width,block_height);
    dim3 grid(grid_width,grid_height);

    ArgmaxBilinearResizeKernel<<<grid,block>>>(src,batch,src_height,src_width,channel,dst,dst_height,dst_width);
}