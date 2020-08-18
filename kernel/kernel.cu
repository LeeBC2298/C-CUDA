#include "kernel.h"
#include <iostream>

using namespace std;

__global__ void kernel(unsigned char* _src_dev, float* _dst_dev, int _src_width, int _src_height, int _dst_width, int _dst_height, float mean, float std){
    double srcXf;
    double srcYf;
    int srcX;
    int srcY;
    int dst_offset;
    int src_offset;

    int y = blockIdx.y*blockDim.y+threadIdx.y;
    int x = blockIdx.x*blockDim.x+threadIdx.x;
    if(x<_dst_width&&y<_dst_height){
        srcXf = x*((float)_src_width/_dst_width);
        srcYf = y*((float)_src_height/_dst_height);
        srcX = (int)srcXf;
        srcY = (int)srcYf;
        dst_offset = (y*_dst_width+x)*3;
        src_offset = (srcY*_src_width+srcX)*3;

        _dst_dev[dst_offset+0] = float((_src_dev[src_offset+2]-mean)/std);
        _dst_dev[dst_offset+1] = float((_src_dev[src_offset+1]-mean)/std);
        _dst_dev[dst_offset+2] = float((_src_dev[src_offset+0]-mean)/std);
    }
}

void preprocess_gpu(unsigned char* src, int ORIGN_W, int ORIGN_H, float* dst, int TARGET_W, int TARGET_H, float mean, float std){
    //cudaSetDevice(9);
    unsigned char *src_dev;
    float *dst_dev;
    cudaMalloc((void**)&src_dev, 3*ORIGN_W*ORIGN_H*sizeof(unsigned char));
    cudaMalloc((void**)&dst_dev, 3*TARGET_W*TARGET_H*sizeof(float));

    /*float time_copy = 0;
    cudaEvent_t start_copy, end_copy;
    cudaEventCreate(&start_copy);
    cudaEventCreate(&end_copy);

    cudaEventRecord(start_copy, 0);*/
    cudaMemcpy(src_dev, src, 3*ORIGN_W*ORIGN_H*sizeof(unsigned char), cudaMemcpyHostToDevice);
    /*cudaEventRecord(end_copy, 0);
    cudaEventSynchronize(start_copy);
    cudaEventSynchronize(end_copy);
    cudaEventElapsedTime(&time_copy, start_copy, end_copy);
    cudaEventDestroy(start_copy);
    cudaEventDestroy(end_copy);
    cout << "gpu copy time: " << time_copy << endl;*/

    //dim3 grid(TARGET_H, TARGET_W);
    //kernel<<<grid,1>>>(src_dev, dst_dev, ORIGN_H, ORIGN_W, TARGET_H, TARGET_W);
    int unit = 16;
    dim3 grid((TARGET_W+unit-1)/unit,(TARGET_H+unit-1)/unit, 1);
    dim3 block(unit, unit, 1);

    /*float time_elapsed = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);*/
    kernel<<<grid,block>>>(src_dev, dst_dev, ORIGN_W, ORIGN_H, TARGET_W, TARGET_H, mean, std);
    cudaDeviceSynchronize();
    /*cudaEventRecord(stop, 0);

    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_elapsed, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop)
    cout << "gpu cal time: " << time_elapsed << endl;;*/

    cudaMemcpy(dst, dst_dev, 3*TARGET_W*TARGET_H*sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(src_dev);
    cudaFree(dst_dev);
}
