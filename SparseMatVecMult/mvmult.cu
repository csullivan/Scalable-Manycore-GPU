#include <wb.h>

#define usingGPU 1
#define ThreadsPerBlock 32

__global__ void spmvCSRKernel(float *out, int *matCols, int *matRows,
                              float *matData, float *vec, int dim) {
  //@@ insert spmv kernel for csr format

  unsigned int idxRow = blockIdx.x * blockDim.x + threadIdx.x;
  if (idxRow<dim){
    int idx=matRows[idxRow];
    float sum=0.0;
    // each thread does it's own row
    // control divergence when one thread
    // finishes and other thread still has
    // more column elements to process
    // should I pad?
    for (int idxCol=matRows[idxRow]; idxCol<matRows[idxRow+1]; idxCol++){
      sum += matData[idx]*vec[matCols[idx++]];
    }
    out[idxRow] = sum;
  }
}


__global__ void spmvJDSKernel(float *out, int *matColStart, int *matCols,
                              int *matRowPerm, int *matRows, float *matData,
                              float *vec, int dim) {
  //@@ insert spmv kernel for jds format

  unsigned int idxRow = blockIdx.x * blockDim.x + threadIdx.x;
  if (idxRow<dim){
    // get the row idx of the unsorted csr matrix
    int idxVec = matRowPerm[idxRow];
    float sum = 0.0;
    for (int idxCol=0; idxCol<matRows[idxRow]; idxCol++){
      int idx = matColStart[idxCol]+idxRow;
      sum += matData[idx]*vec[matCols[idx]];
    }
    out[idxVec] = sum;
  }

}

static void spmvCSR(float *out, int *matCols, int *matRows, float *matData,
                    float *vec, int dim) {

  //@@ invoke spmv kernel for csr format
  dim3 grid((dim-1)/ThreadsPerBlock + 1, 1, 1);
  dim3 block(ThreadsPerBlock, 1, 1);

  spmvCSRKernel <<< grid, block >>> (out,matCols,matRows,matData,vec,dim);

}

static void spmvJDS(float *out, int *matColStart, int *matCols, int *matRowPerm,
                    int *matRows, float *matData, float *vec, int dim) {

  //@@ invoke spmv kernel for jds format

  dim3 grid((dim-1)/ThreadsPerBlock + 1, 1, 1);
  dim3 block(ThreadsPerBlock, 1, 1);

  spmvJDSKernel <<< grid, block >>> (out,matColStart,matCols,matRowPerm,matRows,matData,vec,dim);
}

void spmvCSR_cpu(float *out, int *matCols, int *matRows, float *matData, float *vec, int dim) {

  int idx=0;
  for (int idxRow=0; idxRow<dim; idxRow++){
    idx = matRows[idxRow];
    for (int idxCol=matRows[idxRow]; idxCol<matRows[idxRow+1]; idxCol++){
      out[idxRow] += matData[idx]*vec[matCols[idx++]];
    }
  }

}

void spmvJDS_cpu(float *out, int *matColStart, int *matCols, int *matRowPerm, int *matRows, float *matData, float *vec, int dim){

  int idxRow, idxCol, idxVec;
  int idx=0;
  for (idxRow=0; idxRow<dim; idxRow++){
    idxVec = matRowPerm[idxRow];
    for (idxCol=0; idxCol<matRows[idxRow]; idxCol++){
      idx = matColStart[idxCol]+idxRow;
      out[idxVec] += matData[idx]*vec[matCols[idx]];
    }
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  bool usingJDSQ;
  int *hostCSRCols;
  int *hostCSRRows;
  float *hostCSRData;
  int *hostJDSColStart;
  int *hostJDSCols;
  int *hostJDSRowPerm;
  int *hostJDSRows;
  float *hostJDSData;
  float *hostVector;
  float *hostOutput;
  int *deviceCSRCols;
  int *deviceCSRRows;
  float *deviceCSRData;
  int *deviceJDSColStart;
  int *deviceJDSCols;
  int *deviceJDSRowPerm;
  int *deviceJDSRows;
  float *deviceJDSData;
  float *deviceVector;
  float *deviceOutput;
  int dim, ncols, nrows, ndata;
  int maxRowNNZ;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  usingJDSQ = wbImport_flag(wbArg_getInputFile(args, 0)) == 1;
  hostCSRCols = (int *)wbImport(wbArg_getInputFile(args, 1), &ncols, "Integer");
  hostCSRRows = (int *)wbImport(wbArg_getInputFile(args, 2), &nrows, "Integer");
  hostCSRData = (float *)wbImport(wbArg_getInputFile(args, 3), &ndata, "Real");
  hostVector = (float *)wbImport(wbArg_getInputFile(args, 4), &dim, "Real");

  //hostOutput = (float *)malloc(sizeof(float) * dim);
  hostOutput = (float *)calloc(dim, sizeof(float)); //when just running cpu versions

  wbTime_stop(Generic, "Importing data and creating memory on host");

  if (usingJDSQ) {
    CSRToJDS(dim, hostCSRRows, hostCSRCols, hostCSRData, &hostJDSRowPerm,
       &hostJDSRows, &hostJDSColStart, &hostJDSCols, &hostJDSData);
    maxRowNNZ = hostJDSRows[0];
  }

  if (usingGPU) {

    wbTime_start(GPU, "Allocating GPU memory.");
    if (usingJDSQ) {
      cudaMalloc((void **)&deviceJDSColStart, sizeof(int) * maxRowNNZ);
      cudaMalloc((void **)&deviceJDSCols, sizeof(int) * ndata);
      cudaMalloc((void **)&deviceJDSRowPerm, sizeof(int) * dim);
      cudaMalloc((void **)&deviceJDSRows, sizeof(int) * dim);
      cudaMalloc((void **)&deviceJDSData, sizeof(float) * ndata);
    } else {
      cudaMalloc((void **)&deviceCSRCols, sizeof(int) * ncols);
      cudaMalloc((void **)&deviceCSRRows, sizeof(int) * nrows);
      cudaMalloc((void **)&deviceCSRData, sizeof(float) * ndata);
    }
    cudaMalloc((void **)&deviceVector, sizeof(float) * dim);
    cudaMalloc((void **)&deviceOutput, sizeof(float) * dim);
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    if (usingJDSQ) {
      cudaMemcpy(deviceJDSColStart, hostJDSColStart, sizeof(int) * maxRowNNZ,
           cudaMemcpyHostToDevice);
      cudaMemcpy(deviceJDSCols, hostJDSCols, sizeof(int) * ndata,
           cudaMemcpyHostToDevice);
      cudaMemcpy(deviceJDSRowPerm, hostJDSRowPerm, sizeof(int) * dim,
           cudaMemcpyHostToDevice);
      cudaMemcpy(deviceJDSRows, hostJDSRows, sizeof(int) * dim,
           cudaMemcpyHostToDevice);
      cudaMemcpy(deviceJDSData, hostJDSData, sizeof(float) * ndata,
           cudaMemcpyHostToDevice);
    } else {
      cudaMemcpy(deviceCSRCols, hostCSRCols, sizeof(int) * ncols,
           cudaMemcpyHostToDevice);
      cudaMemcpy(deviceCSRRows, hostCSRRows, sizeof(int) * nrows,
           cudaMemcpyHostToDevice);
      cudaMemcpy(deviceCSRData, hostCSRData, sizeof(float) * ndata,
           cudaMemcpyHostToDevice);
    }
    cudaMemcpy(deviceVector, hostVector, sizeof(float) * dim,
         cudaMemcpyHostToDevice);
    wbTime_stop(GPU, "Copying input memory to the GPU.");

    wbTime_start(Compute, "Performing CUDA computation");
    if (usingJDSQ) {
      spmvJDS(deviceOutput, deviceJDSColStart, deviceJDSCols, deviceJDSRowPerm,
        deviceJDSRows, deviceJDSData, deviceVector, dim);
    } else {
      spmvCSR(deviceOutput, deviceCSRCols, deviceCSRRows, deviceCSRData,
        deviceVector, dim);
    }
    cudaDeviceSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");

    wbTime_start(Copy, "Copying output memory to the CPU");
    cudaMemcpy(hostOutput, deviceOutput, sizeof(float) * dim,
         cudaMemcpyDeviceToHost);
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    cudaFree(deviceCSRCols);
    cudaFree(deviceCSRRows);
    cudaFree(deviceCSRData);
    cudaFree(deviceVector);
    cudaFree(deviceOutput);
    if (usingJDSQ) {
      cudaFree(deviceJDSColStart);
      cudaFree(deviceJDSCols);
      cudaFree(deviceJDSRowPerm);
      cudaFree(deviceJDSRows);
      cudaFree(deviceJDSData);
    }
    wbTime_stop(GPU, "Freeing GPU Memory");

  }else { // if not using GPU, use the CPU versions

    wbTime_start(Compute, "Performing CPU computation");
    if (usingJDSQ) {
      spmvJDS_cpu(hostOutput, hostJDSColStart, hostJDSCols, hostJDSRowPerm, hostJDSRows, hostJDSData, hostVector, dim);
    }
    else {
      spmvCSR_cpu(hostOutput, hostCSRCols, hostCSRRows, hostCSRData, hostVector, dim);
    }
    wbTime_stop(Compute, "Performing CPU computation");
  }

  wbSolution(args, hostOutput, dim);

  free(hostCSRCols);
  free(hostCSRRows);
  free(hostCSRData);
  free(hostVector);
  free(hostOutput);
  if (usingJDSQ) {
    free(hostJDSColStart);
    free(hostJDSCols);
    free(hostJDSRowPerm);
    free(hostJDSRows);
    free(hostJDSData);
  }

  return 0;
}
