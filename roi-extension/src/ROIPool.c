#include <TH/TH.h>
#include <math.h>
#include <float.h>

void ROIPooling_updateOutput(
	THFloatTensor *input,
	THFloatTensor *rois,
	int pooledHeight,
	int pooledWidth,
	float spatialScale,
	THFloatTensor *output,
	THIntTensor *argmaxes)
{
	// Input is the output of the last convolutional layer in the Backbone network, so
	// it should be in the format of NCHW
	THAssert(THFloatTensor_nDimension(input) == 4);

	// ROIs is the set of region proposals to process. It is a 2D Tensor where the first
	// dim is the # of proposals, and the second dim is the proposal itself in the form
	// [batch_index startW startH endW endH]
	THAssert(THFloatTensor_nDimension(rois) == 2);
	THAssert(THFloatTensor_size(rois, 1) == 5);

	int64_t proposals = THFloatTensor_size(rois, 0);
	int64_t inputChannels = THFloatTensor_size(input, 1);
	int64_t inputHeight = THFloatTensor_size(input, 2);
	int64_t inputWidth = THFloatTensor_size(input, 3);

	// Output Tensor is (num_rois, C, pooledHeight, pooledWidth)
	THFloatTensor_resize4d(output, proposals, inputChannels, pooledHeight, pooledWidth);

	// During training, we need to store the argmaxes for the pooling operation, so
	// the argmaxes Tensor should be the same size as the output Tensor
	if (argmaxes != NULL) {
		THIntTensor_resize4d(argmaxes, proposals, inputChannels, pooledHeight, pooledWidth);
	}

	// We use the Tensor's raw buffers in our calculations, so they must be contiguous
	THAssert(THFloatTensor_isContiguous(input));
	THAssert(THFloatTensor_isContiguous(rois));

	float *rawInput = THFloatTensor_data(input);
	int64_t inputChannelStride = inputHeight * inputWidth;
	int64_t inputBatchStride = inputChannels * inputChannelStride;
	float *rawRois = THFloatTensor_data(rois);
	int64_t roiProposalStride = 5;

	float *rawOutput = THFloatTensor_data(output);
	int *rawArgmaxes = THIntTensor_data(argmaxes);
	int64_t outputChannelStride = pooledHeight * pooledWidth;

	// Now that our Tensors are properly sized, we can perform the pooling operation.
	// We iterate over each RoI and perform pooling on each channel in the input, to
	// generate a pooledHeight x pooledWidth output for each RoI
    int64_t i;
	for (i = 0; i < proposals; ++i) {
		int n = (int) rawRois[0];
		int startWidth = roundf(rawRois[1] * spatialScale);
		int startHeight = roundf(rawRois[2] * spatialScale);
		int endWidth = roundf(rawRois[3] * spatialScale);
		int endHeight = roundf(rawRois[4] * spatialScale);

		// TODO: assertions for valid values?
		// TODO: fix malformed ROIs??

		int roiHeight = endHeight - startHeight;
		int roiWidth = endWidth - startWidth;

		// Because the Region of Interest can be of variable size, but our output
		// must always be (pooledHeight x pooledWidth), we need to split the RoI
		// into a pooledHeight x pooledWidth grid of tiles
		float tileHeight = ((float)roiHeight) / ((float)pooledHeight);
		float tileWidth = ((float)roiWidth) / ((float)pooledWidth);

		float *rawInputBatch = rawInput + (n * inputBatchStride);

		// Compute pooling for each of the (pooledHeight x pooledWidth) tiles for each
		// channel in the input
        int ch, ph, pw;
		for (ch = 0; ch < inputChannels; ++ch) {
			for (ph = 0; ph < pooledHeight; ++ph) {
				for (pw = 0; pw < pooledWidth; ++pw) {
					// Need to compute the bounds of this particular tile in the input
					int tileHStart = (int)floorf(((float)ph) * tileHeight);
					int tileWStart = (int)floorf(((float)ph) * tileWidth);
					int tileHEnd = (int)ceilf(((float)ph + 1) * tileHeight);
					int tileWEnd = (int)ceilf(((float)ph + 1) * tileWidth);

					// Add tile offsets to RoI offsets, and clip to input boundaries
					tileHStart = (int) fminf(fmaxf((float) tileHStart + startHeight, 0), inputHeight);
					tileWStart = (int) fminf(fmaxf((float) tileWStart + startWidth, 0), inputWidth);
					tileHEnd = (int) fminf(fmaxf((float) tileHEnd + startHeight, 0), inputHeight);
					tileWEnd = (int) fminf(fmaxf((float) tileWEnd + startWidth, 0), inputWidth);

					int poolIndex = (ph * pooledWidth) + pw;

					// If our pooling region is empty, we set the output to 0, otherwise to
					// the min float so we can calculate the max properly
					int empty = tileHStart >= tileHEnd || tileWStart >= tileWEnd;
					rawOutput[poolIndex] = empty ? 0 : FLT_MIN;

					if (argmaxes != NULL) {
						// Set to -1 so we don't try to backprop to anywhere
						rawArgmaxes[poolIndex] = -1;
					}

					// Iterate over the elements in the tile to find the maximum
                    int th, tw;
					for (th = tileHStart; th < tileHEnd; ++th) {
						for (tw = tileWStart; tw < tileWEnd; ++ tw) {
							int index = (th * inputHeight) + tw;
							if (rawInputBatch[index] > rawOutput[poolIndex]) {
								rawOutput[poolIndex] = rawInputBatch[index];
								if (argmaxes != NULL) {
									rawArgmaxes[poolIndex] = index;
								}
							}
						}
					}
				}
			}

			// Increment raw pointers by channel stride
			rawInputBatch += inputChannelStride;
			rawOutput += outputChannelStride;
			if (argmaxes != NULL) {
				rawArgmaxes += outputChannelStride;
			}
		}

		// Increment RoI raw pointer
		rawRois += roiProposalStride;
	}
}
