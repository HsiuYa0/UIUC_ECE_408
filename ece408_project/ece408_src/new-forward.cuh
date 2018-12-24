
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#include <mxnet/base.h>

#define y4d(i3, i2, i1, i0) y[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
#define x4d(i3, i2, i1, i0) x[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
#define k4d(i3, i2, i1, i0) k[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
#define MIN(a ,b) ((a < b) ? a : b)
#define xunroll3d(i2, i1, i0) x_unroll[(i2) * (C * K * K * H_out * W_out) + (i1) * (H_out * W_out) + i0]
#define kunroll2d(i1, i0) k_unroll[(i1) * (C * K * K) + i0]
#define TILE_WIDTH 16
#define CHANNEL_PARALLELISM 6

namespace mxnet
{
namespace op
{

__global__ void unroll_x(float *x_unroll, const float *x, const int B, const int C, const int H, const int W, const int K) {
	
	/*
		we unroll the input x with shape (B x C x H x W) into (B x C*K*K x Hout*Wout) 
		in this implementation, each thread is responsible for one element in the output
	*/
    
	const int H_out = H - K + 1;	// assuming the stride = 1
    const int W_out = W - K + 1;	// assuming the stride = 1
	
	const int Batch = blockIdx.z * blockDim.z + threadIdx.z;
    const int Row = blockIdx.y * blockDim.y + threadIdx.y;
	const int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if (Batch < B && Row < K * K * C && Col < H_out * W_out) {
		int channel = Row / (K * K);
		int W_shift = (Row % (K * K)) % K;
		int H_shift = (Row % (K * K)) / K;
		xunroll3d(Batch, Row, Col) = x4d(Batch, channel, Col / W_out + H_shift, Col % W_out + W_shift);
	}	
}

__global__ void unroll_k(float *k_unroll, const float *k, const int M, const int C, const int K) {
	
	/*
		unroll the filter k with shape (M x C x K x K) into (M x C*K*K)
		each thread is responsible for one element in the output
	*/
	
	const int Row = blockIdx.y * blockDim.y + threadIdx.y;
	const int Col = blockIdx.x * blockDim.x + threadIdx.x;

	if (Row < M && Col < C * K * K) {
		kunroll2d(Row, Col) = k4d(Row, Col / (K * K), (Col % (K * K)) / K, (Col % (K * K)) % K);
	}
}

__global__ void tiled_matrixMultiply(float *y, const float *k_unroll, const float *x_unroll, const int B, const int M, const int C, const int H, const int W, const int K) {

	/*
		implementation of the tiled matrix multiplication
		assuming that the filter is already in the constant memory kc
		bacause of this, during the tile loading phase, we don't need to load the filter tile into the shared memory

		y: the pointer to the output array with size (B x M x H_out x W_out)
		x_unroll: the pointer to the unrolled input array with size (B, K*K*C, W_out*H_out)
		B: batch_size
		M: number of output channel
		C: number of input channel
		H: height of the input 
		W: width of the input
		K: filter size (assuming squared filter)
	*/
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	
	const int Batch = blockIdx.z * blockDim.z + threadIdx.z;
	const int Row = blockIdx.y * TILE_WIDTH + ty;
	const int Col = blockIdx.x * TILE_WIDTH + tx;
	
	const int H_out = H - K + 1;	// assuming the stride = 1
    const int W_out = W - K + 1;	// assuming the stride = 1
	
	__shared__ float k_unroll_tile[TILE_WIDTH][TILE_WIDTH];
	__shared__ float x_unroll_tile[TILE_WIDTH][TILE_WIDTH];
	
	float result = 0;
	for (int i = 0 ; i < ceil(K * K * C / (1.0 * TILE_WIDTH)); i++) {
		
		// tile loading phase	
		int currCol = TILE_WIDTH * i + tx;
		if (Row < M && currCol < K * K * C) {
			k_unroll_tile[ty][tx] = kunroll2d(Row, currCol);
		} else {
			k_unroll_tile[ty][tx] = 0;
		}

		int currRow = TILE_WIDTH * i + ty;
		if (Batch < B && currRow < K * K * C && Col < H_out * W_out) {
			x_unroll_tile[ty][tx] = xunroll3d(Batch, currRow, Col);
		} else {
			x_unroll_tile[ty][tx] = 0;
		}
		
		__syncthreads(); // make sure the loading is completed
		
		// calculation phase
		for (int j = 0; j < TILE_WIDTH; j++) {
			result += k_unroll_tile[ty][j] * x_unroll_tile[j][tx];
		}
		
		__syncthreads(); // make sure the shared memory is consumed
	}
	
	if (Batch < B && Row < M && Col < H_out * W_out) {
		y4d(Batch, Row, Col / W_out, Col % W_out) = result;
	}
}

__global__ void fusion_kernel(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K)
{

    /*
		implement kernel fusion for unrolling and matrix multiplication
		we only do parallelism in input images
		
		we unroll the filter & input during the loading phase
		during unrolling, since we have already put the filter in the constant memory, 
		there is no need to put it in the shared memory
		the input should be unrolled into a K * K * C by Hout * W_out matrix
		
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

	
    const int H_out = H - K + 1;	// assuming the stride = 1
    const int W_out = W - K + 1;	// assuming the stride = 1

	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	
	const int Row = blockIdx.y * TILE_WIDTH + ty; // Row corresponds to the index of filter (M)
	const int Col = blockIdx.x * TILE_WIDTH + tx; // Col corresponds to the index of output element

	__shared__ float k_tile[TILE_WIDTH][TILE_WIDTH];
	__shared__ float x_tile[TILE_WIDTH][TILE_WIDTH];
	
	float result = 0;
	for (int i = 0; i < ceil(K * K * C / (1.0 * TILE_WIDTH)); i++) {
		
		// load the tile with unrolling 
		int currCol = TILE_WIDTH * i + threadIdx.x;
		int k_channel = currCol / (K * K);
		int k_row = (currCol % (K * K)) / K;
		int k_col = (currCol % (K * K)) % K;
		
		// since k_row & k_col will always be in the valid range, we don't neet to check their validity
		if (Row < M && currCol < C * K * K) {
			k_tile[ty][tx] = k4d(Row, k_channel, k_row, k_col);
		} else {
			k_tile[ty][tx] = 0;
		}
		int currRow = TILE_WIDTH * i + ty;
		int x_channel = currRow / (K * K);
		int x_row_shift = (currRow % (K * K)) / K;
		int x_col_shift = (currRow % (K * K)) % K;
		
		if (currRow < C * K * K && Col < H_out * W_out) {
			x_tile[ty][tx] = x4d(blockIdx.z, x_channel, Col / W_out + x_row_shift, Col % W_out + x_col_shift);
		} else {
			x_tile[ty][tx] = 0;
		}

		__syncthreads(); // make sure the loading is completed
		
		for (int j = 0; j < TILE_WIDTH; j++) {
			result += k_tile[ty][j] * x_tile[j][tx];
			/*
			int col_index = TILE_WIDTH * i + j;
			if (Row < M && (col_index) < K * K * C) { 
				result += kc[Row * K * K * C + col_index] * x_tile[j][tx];
			}
			*/
		}
		__syncthreads(); // make sure the tiles are consumed
	}
	
	if (Row < M && Col < W_out * H_out) {
		y4d(blockIdx.z, Row, Col / W_out, Col % W_out) = result;
	}

}

__global__ void fusion_kernel_inputChannel_parallelism(float *y, const float *x, const float *k, const int B, const int M, const int C, const int H, const int W, const int K) {
	const int tx = threadIdx.x;
	const int ty = threadIdx.y;
	const int H_out = H - K + 1;
	const int W_out = W - K + 1;
	const int Channel_start = blockIdx.y * CHANNEL_PARALLELISM * K * K;
	const int Channel_end = (blockIdx.y + 1) * CHANNEL_PARALLELISM * K * K;
	const int W_grid = ceil(H_out * W_out / (1.0 * TILE_WIDTH));
	const int Row = blockIdx.z / W_grid * TILE_WIDTH + threadIdx.y;
	const int Col = blockIdx.z % W_grid * TILE_WIDTH + threadIdx.x;

	__shared__ float k_tile[TILE_WIDTH][TILE_WIDTH];
	__shared__ float x_tile[TILE_WIDTH][TILE_WIDTH];

	float result = 0;
	for (int i = 0; i < ceil(K * K * CHANNEL_PARALLELISM / (1.0 * TILE_WIDTH)); i++) {
		// loading phase
		int currCol = Channel_start + TILE_WIDTH * i + threadIdx.x;
		int k_channel = currCol / (K * K);
		int k_row = (currCol % (K * K)) / K;
		int k_col = (currCol % (K * K)) % K;
		if (Row < M && currCol < MIN(Channel_end, K * K * C)) {
			k_tile[ty][tx] = k4d(Row, k_channel, k_row, k_col);
		} else {
			k_tile[ty][tx] = 0;
		}

		int currRow = Channel_start + TILE_WIDTH * i + threadIdx.y;
		int x_channel = currRow / (K * K);
		int x_row_shift = (currRow % (K * K)) / K;
		int x_col_shift = (currRow % (K * K)) % K;
		
		if (currRow < MIN(Channel_end, K * K * C) && Col < H_out * W_out) {
			x_tile[ty][tx] = x4d(blockIdx.x, x_channel, Col / W_out + x_row_shift, Col % W_out + x_col_shift);
		} else {
			x_tile[ty][tx] = 0;
		}

		__syncthreads(); // make sure the loading is completed
		
		for (int j = 0; j < TILE_WIDTH; j++) {
			result += k_tile[ty][j] * x_tile[j][tx];
		}
		
		__syncthreads(); // make sure the tiles are consumed
	}

	if (Row < M && Col < H_out * W_out) {
		atomicAdd(&(y4d(blockIdx.x, Row, Col / W_out, Col % W_out)), result);	
	}

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
	
    // Set the kernel dimensions
	int H_out = H - K + 1;
	int W_out = W - K + 1;
	
	
	// // call the unroll kernel
	// float *k_unroll;
	// cudaMalloc((void **) &k_unroll, M * K * K * C * sizeof(float));
	// dim3 gridDim_k_unroll(ceil(K * K * C / (1.0 * TILE_WIDTH)), ceil(M / (1.0 * TILE_WIDTH)), 1);
	// dim3 blockDim_k_unroll(TILE_WIDTH, TILE_WIDTH, 1);
	// unroll_k<<<gridDim_k_unroll, blockDim_k_unroll>>>(k_unroll, w.dptr_, M, C, K); 

	// float *x_unroll;
	// cudaMalloc((void **) &x_unroll, B * K * K * C * H_out * W_out * sizeof(float));
	// dim3 gridDim_unroll(ceil(H_out * W_out / (1.0 * TILE_WIDTH)), ceil(K * K * C / (1.0 * TILE_WIDTH)), B);
	// dim3 blockDim_unroll(TILE_WIDTH, TILE_WIDTH, 1);
	// unroll_x<<<gridDim_unroll, blockDim_unroll>>>(x_unroll, x.dptr_, B, C, H, W, K);
	
	// // call the tiled matrix multiplication kernel
	// dim3 gridDim_mm(ceil(H_out * W_out / (1.0 * TILE_WIDTH)), ceil(M / (1.0 * TILE_WIDTH)), B);
	// dim3 blockDim_mm(TILE_WIDTH, TILE_WIDTH, 1);
	// tiled_matrixMultiply<<<gridDim_mm, blockDim_mm>>>(y.dptr_, k_unroll, x_unroll, B, M, C, H, W, K);
	
		
	// call the fusion kernel
	dim3 gridDim(ceil(H_out * W_out / (1.0 * TILE_WIDTH)), ceil(M / (1.0 * TILE_WIDTH)), B);
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    fusion_kernel<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
	

	// call the fusion kernel
	// dim3 gridDim(B, ceil(C / (1.0 * CHANNEL_PARALLELISM)), ceil(H_out * W_out / (1.0 * TILE_WIDTH)) * ceil(M / (1.0 * TILE_WIDTH)));
 //    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
 //    fusion_kernel_inputChannel_parallelism<<<gridDim, blockDim>>>(y.dptr_,x.dptr_,w.dptr_, B,M,C,H,W,K);
	
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

#undef y4d
#undef x4d
#undef k4d
#undef xunroll3d
#undef kunroll2di
#undef MIN
#endif
