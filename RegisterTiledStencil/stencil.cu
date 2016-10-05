#include <wb.h>

#define wbCheck(stmt)                                                     \
do {                                                                      \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

void stencil_cpu(char *_out, char *_in, int width, int height, int depth) {

#define out(i, j, k) _out[((i)*width + (j)) * depth + (k)]
#define in(i, j, k) _in[((i)*width + (j)) * depth + (k)]

  for (int i = 1; i < height - 1; ++i) {
    for (int j = 1; j < width - 1; ++j) {
      for (int k = 1; k < depth - 1; ++k) {
        out(i, j, k) = in(i, j, k + 1) + in(i, j, k - 1) +
          in(i, j + 1, k) + in(i, j - 1, k) +
          in(i + 1, j, k) + in(i - 1, j, k) - 6 * in(i, j, k);
      }
    }
  }
#undef out
#undef in
}

#define idx3d(i,j,k) ((i)*width + (j)) * depth + (k)
#define tile_size 16


__device__ float clamp(float val, int min, int max) {
  return (val > max) ? max : (val < min) ? min : val;
}


__global__ void stencil(float *output, float *input, int width, int height,
      int depth) {
  //@@ INSERT CODE HERE
  int i = blockIdx.z*blockDim.z + threadIdx.z;
  int j = blockIdx.y*blockDim.y + threadIdx.y;
  __shared__ float zy_2d_cache[tile_size][tile_size];
  float prev = input[idx3d(i,j,0)];
  float current = input[idx3d(i,j,1)];
  float next = input[idx3d(i,j,2)];
  zy_2d_cache[threadIdx.z][threadIdx.y] = current;
  __syncthreads();

  if (i < height && j < width) {
    for(int k = 1; k < depth-1; k++) {

      // I start with the lower end of the array. One needs to check that the thread index is not on the edge
      // if it is, and the block is the lowest one, then the halo value is 0, otherwise it's non zero.
      float bottom = (threadIdx.z > 0) ? zy_2d_cache[threadIdx.z - 1][threadIdx.y] : (blockIdx.z > 0) ? input[idx3d(i-1,j,k)] : 0;
      float left = (threadIdx.y > 0) ? zy_2d_cache[threadIdx.z][threadIdx.y - 1] : (blockIdx.y > 0) ? input[idx3d(i,j-1,k)] : 0;
      // just like above, we now do the same thing with the upper end of the image array. if the thread index is
      // not at the top, the we load from shared memory, if it is, then we check that it's not the global top. if it is, then it's 0, else its a
      // a globally loaded halo value.
      float top = (threadIdx.z < blockDim.z-1) ? zy_2d_cache[threadIdx.z + 1][threadIdx.y] : (blockIdx.z < gridDim.z-1) ? input[idx3d(i+1,j,k)] : 0;
      float right = (threadIdx.y < blockDim.y-1) ? zy_2d_cache[threadIdx.z][threadIdx.y+1] : (blockIdx.y < gridDim.y-1) ? input[idx3d(i,j+1,k)] : 0;
      float temp = prev + next + top + bottom + right + left - 6*current;


      output[idx3d(i,j,k)] =  clamp(temp,0,255);
      if (i==0 || i == height-1 ||  j ==0 || j==width-1) {  output[idx3d(i,j,k)] = 0.; }

      prev = current;
      __syncthreads();
      zy_2d_cache[threadIdx.z][threadIdx.y]=next;
      __syncthreads();
      current = next;
      next = input[idx3d(i,j,k+2)];
    }

  }

}


__global__ void stencil_simple(float *output, float *input, int width, int height,
            int depth) {

#define out(i, j, k) output[((i)*width + (j)) * depth + (k)]
#define in(i, j, k) input[((i)*width + (j)) * depth + (k)]

  int i = blockIdx.z*blockDim.z + threadIdx.z;
  int j = blockIdx.y*blockDim.y + threadIdx.y;

  if (i < height && j < width) {
    for (int k = 1; k < depth - 1; ++k) {
      float temp = in(i, j, k + 1) + in(i, j, k - 1) +
        in(i, j + 1, k) + in(i, j - 1, k) +
        in(i + 1, j, k) + in(i - 1, j, k) - 6 * in(i, j, k);


      out(i,j,k) = clamp(temp,0,255);
      if (i==0 || i == height-1 ||  j ==0 || j==width-1) {  out(i,j,k) = 0.; }
    }
  }
#undef out
#undef in
}


static void launch_stencil(float *deviceOutputData, float *deviceInputData,
         int width, int height, int depth) {
  //@@ INSERT CODE HERE
  dim3 grid(1,(width-1)/tile_size + 1,(height-1)/tile_size +1);
  dim3 block(1,tile_size,tile_size);
  stencil<<<grid,block>>>(deviceOutputData,deviceInputData,width,height,depth);
}




int main(int argc, char *argv[]) {
  wbArg_t arg;
  int width;
  int height;
  int depth;
  char *inputFile;
  wbImage_t input;
  wbImage_t output;
  float *hostInputData;
  float *hostOutputData;
  float *deviceInputData;
  float *deviceOutputData;

  arg = wbArg_read(argc, argv);

  inputFile = wbArg_getInputFile(arg, 0);

  input = wbImport(inputFile);

  width = wbImage_getWidth(input);
  height = wbImage_getHeight(input);
  depth = wbImage_getChannels(input);

  output = wbImage_new(width, height, depth);
  wbLog(TRACE, "Image size: ", width, " x ", height, " x ", depth);

  hostInputData = wbImage_getData(input);
  hostOutputData = wbImage_getData(output);

  wbTime_start(GPU, "Doing GPU memory allocation");
  cudaMalloc((void **)&deviceInputData,
       width * height * depth * sizeof(float));
  cudaMalloc((void **)&deviceOutputData,
       width * height * depth * sizeof(float));
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  cudaMemcpy(deviceInputData, hostInputData,
       width * height * depth * sizeof(float),
       cudaMemcpyHostToDevice);
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  launch_stencil(deviceOutputData, deviceInputData, width, height, depth);
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutputData, deviceOutputData,
       width * height * depth * sizeof(float),
       cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

  wbSolution(arg, output);

  cudaFree(deviceInputData);
  cudaFree(deviceOutputData);

  wbImage_delete(output);
  wbImage_delete(input);

  return 0;
}
