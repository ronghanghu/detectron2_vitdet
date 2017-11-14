#include <THC/THC.h>

// this symbol will be resolved automatically from PyTorch libs
extern THCState *state;

void ROIPooling_updateOutput(
    THCState *state,
	THCudaTensor *input,
	THCudaTensor *rois,
	int pooledHeight,
	int pooledWidth,
	float spatialScale,
	THCudaTensor *output,
	THCudaIntTensor *argmaxes) {

}
