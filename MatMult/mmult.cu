#include <wb.h>

#define wbCheck(stmt)                                                          \
  do {                                                                         \
    cudaError_t err = stmt;                                                    \
    if (err != cudaSuccess) {                                                  \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                              \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));           \
      return -1;                                                               \
    }                                                                          \
  } while (0)


// matrix A is in column major format
#define idxColumnMajor(idxrow,idxcol,nrow,ncol) idxcol*nrow+idxrow

// matrix B is in row major format
#define idxRowMajor(idxrow,idxcol,nrow,ncol) idxrow*ncol+idxcol


#define TILE_SIZE_A 32
#define TILE_SIZE_B 8

// Compute C = A * B
__global__ void matrixMultiply_kernel(float *A, float *B, float *C,
                                      int numARows, int numAColumns,
                                      int numBRows, int numBColumns,
                                      int numCRows, int numCColumns) {
  //@@ Insert code to implement matrix multiplication here
  //@@ You have to perform register tiling for this MP
  __shared__ float sB_tile[TILE_SIZE_B];
  float C_tile[TILE_SIZE_B] = {};

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int tx = threadIdx.x;
  int ty = threadIdx.y;

  // thread coarsening: for each input matrix element of A (thread), we TILE_SIZE_B inner products
  int idxRow = by * TILE_SIZE_A + ty;
  // shared memory tiling of input matrix B
  int idxCol = bx * TILE_SIZE_B + tx;
  // memory coalescing on both A and B (since A is transposed)
  if (idxRow<numARows && idxCol<numBColumns){

    for(int k=0;k<numBRows;k++){

      // load tile of B into gpu shared memory
      if (tx<TILE_SIZE_B){
        int idxB = idxRowMajor(k,idxCol,numBRows,numBColumns);
        sB_tile[tx] = B[idxB];
      }
      __syncthreads();

      //perform matrix mult with B shared tile and save it into local register array for C
      int idxA = idxColumnMajor(idxRow,k,numARows,numAColumns);
      float A_element = A[idxA]; // store in register
      // loop over tile of B which is in shared memory
      // this is thread coarsening on
      for (int idxB=0; idxB<TILE_SIZE_B; idxB++){
        C_tile[idxB] += A_element*sB_tile[idxB];
      }

      __syncthreads();
    }

    // save changes of local output tile into global C matrix
    for (int idxTile=0; idxTile<TILE_SIZE_B; idxTile++){
      idxCol = bx * TILE_SIZE_B + idxTile;
      // make sure not to exceed number of col. in B otherwise we
      // will start writing into the next row unintentionally
      if (idxCol<numBColumns){
        int idxC = idxRowMajor(idxRow,idxCol,numCRows,numCColumns);
        C[idxC] = C_tile[idxTile];
      }
    }

  }
}

static void matrixMultiply(float *A, float *B, float *C, int numARows,
                           int numAColumns, int numBRows, int numBColumns,
                           int numCRows, int numCColumns) {
  //@@ Insert code to launch matrix multiplication
  dim3 grid((numBColumns-1)/TILE_SIZE_B + 1, (numARows-1)/TILE_SIZE_A + 1, 1);
  dim3 block(TILE_SIZE_B, TILE_SIZE_A, 1);

  matrixMultiply_kernel <<< grid, block >>>
    (A,B,C,
     numARows,numAColumns,
     numBRows,numBColumns,
     numCRows,numCColumns);
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numAColumns,
                            &numARows);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  numCRows = numARows;
  numCColumns = numBColumns;
  hostC = (float *)malloc(sizeof(float) * numCRows * numCColumns);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  cudaMalloc((void **)&deviceA, sizeof(float) * numARows * numAColumns);
  cudaMalloc((void **)&deviceB, sizeof(float) * numBRows * numBColumns);
  cudaMalloc((void **)&deviceC, sizeof(float) * numCRows * numCColumns);
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  cudaMemcpy(deviceA, hostA, sizeof(float) * numARows * numAColumns,
             cudaMemcpyHostToDevice);
  cudaMemcpy(deviceB, hostB, sizeof(float) * numBRows * numBColumns,
             cudaMemcpyHostToDevice);
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  wbTime_start(Compute, "Performing CUDA computation");
  matrixMultiply(deviceA, deviceB, deviceC, numARows, numAColumns,
                 numBRows, numBColumns, numCRows, numCColumns);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  cudaMemcpy(hostC, deviceC, sizeof(float) * numCRows * numCColumns,
             cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);

  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}



