void ROIPooling_updateOutput(
	THFloatTensor *input,
	THFloatTensor *rois,
	int pooledHeight,
	int pooledWidth,
	float spatialScale,
	THFloatTensor *output,
	THIntTensor *argmaxes);
