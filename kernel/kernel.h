#include "cuda_runtime.h"

void preprocess_gpu(unsigned char* src, int ORIGN_W, int ORIGN_H, float*dst, int TARGET_W, int TARGHT_H, float mean, float std);
