#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
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

#define idx3d(i, j, k) (k*height + j) * width + i
//#define idx3d(i,j,k) ((i)*width + (j)) * depth + (k)
#define tile_size 32

__global__ void stencil(float *output, float *input, int width, int height,
                        int depth) {
  //@@ INSERT CODE HERE
  unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
  unsigned int k = blockIdx.z*blockDim.z + threadIdx.z;

  __shared__ float xz_2d_cache[tile_size][tile_size];
  float prev = input[idx3d(i,0,k)];
  float current = input[idx3d(i,1,k)];
  xz_2d_cache[i][k] = current;
  float next = input[idx3d(i,2,k)];

  for(auto j = 1u; j < height-1; j++) {
    output[idx3d(i,j,k)] = prev + next +
      (threadIdx.x > 0) ? xz_2d_cache[threadIdx.x - 1][threadIdx.z] : input[idx3d(i-1,j,k)] +
      (threadIdx.x < blockDim.x) ? xz_2d_cache[threadIdx.x + 1][threadIdx.z] : input[idx3d(i+1,j,k)] +
      (threadIdx.z > 0) ? xz_2d_cache[threadIdx.x][threadIdx.z - 1] : input[idx3d(i,j,k-1)] +
      (threadIdx.z < blockDim.z) ? xz_2d_cache[threadIdx.x][threadIdx.z+1] : input[idx3d(i,j,k+1)] -
      6*current;

    prev = current;
    xz_2d_cache[threadIdx.x][threadIdx.z]=next;
    current = next;
    __syncthreads();
    next = input[idx3d(i,j+1,k)];
    __syncthreads();
  }

}

static void launch_stencil(float *deviceOutputData, float *deviceInputData,
                           int width, int height, int depth) {
  //@@ INSERT CODE HERE
  auto len = width*height*depth;
  auto nThreads = tile_size*tile_size;
  auto nBlocks = (len+nThreads-1)/nThreads;
  stencil<<<nBlocks,nThreads>>>(deviceOutputData,deviceInputData,width,height,depth);


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
