
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

namespace mxnet
{
namespace op
{

constexpr int TILE_WIDTH = 8;
__constant__ float kernel[14112];
// move const float* k to constant memory
__global__ void forward_kernel(float *y, const float *x, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
		the convolution forward kernel using strategy 3, 
		y: the pointer to the output array with size (B x M x H_out x W_out)
		x: the pointer to the input array with size (B x C x H x W)
		k: the pointer to the filter (M x C x K x K)
		B: batch_size
		M: number of output channel
		C: number of input channel
		H: height of the input 
		W: width of the input
		K: filter size (assuming squared filter)
	*/

	#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
	#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
	#define k4d(i3, i2, i1, i0) kernel[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
	#define MIN( a, b ) ( (a < b) ? a : b )

	// use stratege 3 
	__shared__ float x_tile[TILE_WIDTH][TILE_WIDTH];
	
    const int H_out = H - K + 1;	// assuming the stride = 1
    const int W_out = W - K + 1;	// assuming the stride = 1
	const int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));
	
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	
	const int bx = blockIdx.x;	// bx corresponds to batch
	const int by = blockIdx.y;	// by corresponds to output channel 
	// get the corresponding x, y positions in the output feature map for the thread
	const int tile_start_x = blockIdx.z % W_grid * TILE_WIDTH;
	const int tile_end_x = tile_start_x + TILE_WIDTH;
	const int tile_start_y = blockIdx.z / W_grid * TILE_WIDTH;
	const int tile_end_y = tile_start_y + TILE_WIDTH;
	const int w = tile_start_x + tx;
	const int h = tile_start_y + ty;

	float results = 0;
	for (int c = 0; c < C; c++) {
		
		// load the data into the shared memory
		if (w < W_out && h < H_out) {
			// the coordinate between the input and output feature map is shifted by K/2
			x_tile[ty][tx] = x4d(bx, c, h + K/2, w + K/2); 
		} else {
			x_tile[ty][tx] = 0;
		}
		__syncthreads(); // make sure the loading is done
		

		// shift the output coordinate to input coordinate
		int input_start_x = w;	// w - K/2 + K/2;
		int input_start_y = h;	// h - K/2 + K/2;
		int input_tile_start_x = tile_start_x + K/2;
		int input_tile_end_x = tile_end_x + K/2;
		int input_tile_start_y = tile_start_y + K/2;
		int input_tile_end_y = tile_end_y + K/2;
		
		for (int i = 0; i < K; i++) {
			for (int j = 0; j < K; j++) {
				int input_idx_x = w + i;
				int input_idx_y = h + j;
				if (input_idx_x < W && input_idx_y < H) {
					if (input_idx_x >= input_tile_start_x && input_idx_x < MIN(input_tile_end_x, W_out + K/2) \
						&& input_idx_y >= input_tile_start_y && input_idx_y < MIN(input_tile_end_y, H_out + K/2)) {
						results += x_tile[ty + j - K/2][tx + i - K/2] * k4d(by, c, j, i);
					} else {
						// load the elements of x directly from global memory
						results += x4d(bx, c, input_idx_y, input_idx_x) * k4d(by, c, j, i);
					}
				}
			}
		}
		__syncthreads(); // make sure all threads finish its computation before loading new data into shared memory
	}

	if (w < W_out && h < H_out) {
		y4d(bx, by, h, w) = results;
	}

	#undef y4d
	#undef x4d
	#undef k4d
	#undef MIN
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template <>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w)
{

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];	// batch size
    const int C = x.shape_[1];	// input channel
    const int H = x.shape_[2];	// height of the input feature map
    const int W = x.shape_[3];	// width of the input feature map
	const int M = y.shape_[1];	// output channel
    const int K = w.shape_[3];	// the size of the filter (assuming squared filter)
	//printf("K[0]=%d\nK[1]=%d\nK[2]=%d\nK[3]=%d\n", w.shape_[0], w.shape_[1], w.shape_[2], w.shape_[3]);
	
    
	cudaMemcpyToSymbol(kernel, w.dptr_, sizeof(float) * w.shape_[0] * w.shape_[1] * w.shape_[2] * w.shape_[3], 0, cudaMemcpyHostToDevice);
    // Set the kernel dimensions
	int H_out = H - K + 1;
	int W_out = W - K + 1;

	int H_grid = ceil(H_out / (1.0 * TILE_WIDTH));
	int W_grid = ceil(W_out / (1.0 * TILE_WIDTH));

	dim3 gridDim(B, M, H_grid * W_grid);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);

	//cudaMemcpyToSymbol(kernel, w.dptr_, sizeof(float) * K * K, 0, cudaMemcpyHostToDevice);
    // Call the kernel
    forward_kernel<<<gridDim, blockDim>>>(y.dptr_, x.dptr_, B,M,C,H,W,K);
    // Use MSHADOW_CUDA_CALL to check for CUDA runtime errors.
    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

}

/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template <typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w)
{
    CHECK_EQ(0,1) << "Remove this line and replace it with your implementation.";
}
}
}

#endif
