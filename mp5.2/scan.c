// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, float *sum, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  
  __shared__ int T[2*BLOCK_SIZE];
  for(int k=0; k<2; ++k){
    int tid = threadIdx.x + (blockIdx.x * blockDim.x * 2) + k * blockDim.x;
    if(tid < len){
      T[threadIdx.x + k*blockDim.x] = input[tid];
    }else{
      T[threadIdx.x + k*blockDim.x] = 0;
    }  
  }
  __syncthreads();
  
  // reduction
  for(int stride = 1; stride <= BLOCK_SIZE; stride *= 2){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index < 2*BLOCK_SIZE && (index-stride) >= 0){
      T[index] += T[index-stride];
    }
  }
    
  // post_scan
  for(int stride = BLOCK_SIZE/2 ; stride > 0; stride /= 2){
      __syncthreads();
      int index = (threadIdx.x + 1) * stride * 2 - 1;
      if ((index+stride) < 2*BLOCK_SIZE)
          T[index+stride] += T[index];
  }
  
  // copy to output
  __syncthreads();
  for(int k=0; k<2; ++k){
    int tid = threadIdx.x + (blockIdx.x * blockDim.x * 2) + k * blockDim.x;
    if(tid < len){
      output[tid] = T[threadIdx.x + k*blockDim.x];
    } 
  }
  // store partial sum
  if(threadIdx.x == 0)
    sum[blockIdx.x] = T[2*BLOCK_SIZE-1];
}

__global__ void add(float *output, float *sum, int len){
  __shared__ float increment;
  if (threadIdx.x == 0)
    increment = blockIdx.x == 0 ? 0 : sum[blockIdx.x - 1];
  __syncthreads();
  
  for(int k = 0; k < 2; ++k){
    int tid = (blockIdx.x * blockDim.x * 2) + threadIdx.x + (k * BLOCK_SIZE);
    if(tid < len){
      output[tid] += increment;
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceSumBuffer;
  float *deviceSum;
  float *tmpBuffer;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  int numBlocks = ceil((numElements*1.0) / (BLOCK_SIZE*2));
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceSumBuffer, numBlocks * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceSum, numBlocks * sizeof(float)));
  wbCheck(cudaMalloc((void **)&tmpBuffer, 1 * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 DimGrid(numBlocks, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);
  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  
  // scan input array
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, deviceSumBuffer, numElements);
  cudaDeviceSynchronize();
  
  // scan aux array
  dim3 singleDimGrid(1, 1, 1);
  scan<<<singleDimGrid, DimBlock>>>(deviceSumBuffer, deviceSum, tmpBuffer, numBlocks);
  cudaDeviceSynchronize();
  
  // add
  add<<<DimGrid, DimBlock>>>(deviceOutput, deviceSum, numElements);
  cudaDeviceSynchronize();
  
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
