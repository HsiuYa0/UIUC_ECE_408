// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
typedef unsigned char uint8_t;
typedef unsigned int  uint_t;

//@@ insert code here
__global__ void castimg(float * input, uint8_t* output, int width, int height){
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(y < height && x < width){
    int idx = blockIdx.z * (width * height) + y * (width) + x;
    output[idx] = (uint8_t) ((HISTOGRAM_LENGTH - 1) * input[idx]);
  }
}

__global__ void rgb2gray(uint8_t* input, uint8_t* output, int width, int height){
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if(y < height && x < width){
    int idx = y * (width) + x;
    uint8_t r = input[3 * idx];
    uint8_t g = input[3 * idx + 1];
    uint8_t b = input[3 * idx + 2];
    output[idx] = (uint8_t) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void gray2histogram(uint8_t* input, uint_t* output, int width, int height){
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  /*
  if(y < height && x < width){
    int idx = y * (width) + x;
    uint8_t val = input[idx];
    atomicAdd(&(output[val]), 1);
  }*/
  
  __shared__ uint_t his[HISTOGRAM_LENGTH];
  int tIdx = threadIdx.x + threadIdx.y * blockDim.x;
  if (tIdx < HISTOGRAM_LENGTH) {
    his[tIdx] = 0;
  }
  __syncthreads();
  
  if (x < width && y < height) {
    int idx = y * (width) + x;
    uint8_t val = input[idx];
    atomicAdd(&(his[val]), 1);
  }

  __syncthreads();
  if (tIdx < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[tIdx]), his[tIdx]);
  }
}

__global__ void histogram2CDF(uint_t* input, float* output, int width, int height){
  __shared__ uint_t cdf[HISTOGRAM_LENGTH];
  int id = threadIdx.x;
  cdf[id] = input[id];
  __syncthreads();
  
  // reduction
  for(int stride = 1; stride <= HISTOGRAM_LENGTH/2; stride *= 2){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if(index < HISTOGRAM_LENGTH && (index-stride) >= 0){
      cdf[index] += cdf[index-stride];
    }
  }
  
  // post scan
  for(int stride = HISTOGRAM_LENGTH/4 ; stride > 0; stride /= 2){
      __syncthreads();
      int index = (threadIdx.x + 1) * stride * 2 - 1;
      if ((index+stride) < HISTOGRAM_LENGTH)
          cdf[index+stride] += cdf[index];
  }
  __syncthreads();
  output[id] = cdf[id] / ((float)(width * height));
}

__global__ void equalization(uint8_t* img, float* cdf, int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < width && y < height){
    int idx = blockIdx.z * (width * height) + y * (width) + x;
    float v = 255*(cdf[img[idx]] - cdf[0])/(1.0 - cdf[0]);
    v = min(max(v, 0.0), 255.0);
    img[idx] = (uint8_t) v;
  }
}

__global__ void uint2float(uint8_t* input, float* output, int width, int height){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < width && y < height){
    int idx = blockIdx.z * (width * height) + y * (width) + x;
    output[idx] = (float) (input[idx] / 255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float *deviceImgFloat;
  uint8_t *deviceImgUint;
  uint8_t *deviceImgGray;
  uint_t *deviceHistogram;
  float *deviceCDF; 

  args = wbArg_read(argc, argv); /* parse the input arguments */
  inputImageFile = wbArg_getInputFile(args, 0);
  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);//get image data 
  hostOutputImageData = wbImage_getData(outputImage); 
  wbTime_stop(Generic, "Importing data and creating memory on host");
  
  //@@ insert code here
  cudaMalloc((void**) &deviceImgFloat,       imageWidth * imageHeight * imageChannels * sizeof(float));
  cudaMalloc((void**) &deviceImgUint,        imageWidth * imageHeight * imageChannels * sizeof(uint8_t));
  cudaMalloc((void**) &deviceImgGray,        imageWidth * imageHeight *                 sizeof(uint8_t));
  cudaMalloc((void**) &deviceHistogram,      HISTOGRAM_LENGTH *                         sizeof(uint_t));
  cudaMemset((void *) deviceHistogram, 0,    HISTOGRAM_LENGTH *                         sizeof(uint_t));
  cudaMalloc((void**) &deviceCDF,            HISTOGRAM_LENGTH *                         sizeof(float));
  
  cudaMemcpy(deviceImgFloat, hostInputImageData, 
             imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice);
  dim3 dimGrid;
  dim3 dimBlock;
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  castimg<<<dimGrid, dimBlock>>>(deviceImgFloat, deviceImgUint, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);
  rgb2gray<<<dimGrid, dimBlock>>>(deviceImgUint, deviceImgGray, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), 1);
  dimBlock = dim3(32, 32, 1);
  wbTime_start(Generic, "Privatization");
  gray2histogram<<<dimGrid, dimBlock>>>(deviceImgGray, deviceHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  wbTime_stop(Generic, "Privatization");
  
  
  dimGrid  = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
  histogram2CDF<<<dimGrid, dimBlock>>>(deviceHistogram, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  equalization<<<dimGrid, dimBlock>>>(deviceImgUint, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  
  dimGrid  = dim3(ceil(imageWidth/32.0), ceil(imageHeight/32.0), imageChannels);
  dimBlock = dim3(32, 32, 1);
  uint2float<<<dimGrid, dimBlock>>>(deviceImgUint, deviceImgFloat, imageWidth, imageHeight);
  cudaDeviceSynchronize();
  
  cudaMemcpy(hostOutputImageData, deviceImgFloat,
             imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceImgFloat);
  cudaFree(deviceImgUint);
  cudaFree(deviceImgGray);
  cudaFree(deviceHistogram);
  cudaFree(deviceCDF);

  return 0;
}
