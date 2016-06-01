//nvcc -arch=sm_35 -rdc=true main.cu /usr/lib/libndt.a -lcudadevrt 

#include <stdio.h>
#include "ndt.h"

__global__ void small_kernel(float* A){
  int tid = blockIdx.x*blockDim.x + threadIdx.x; 
  A[tid] = 0.0;
  return;
}

__global__ void run100(float* A, int num){
  for(int i=0; i<num; i++){
    small_kernel<<<4,1024>>>(A);
  }
  cudaDeviceSynchronize();
}

int main(){
  /* data initialization */
  float *A;
  int num = 100;
  cudaSetDevice(0);
  NDT_ERR
  cudaMalloc(&A, 1920*1080*sizeof(float));
  NDT_ERR
  long long start;
  float time1,time2;
  ndt_cpu_timer_start(&start);
  /* small_kernel num times */
  for(int i=0; i<num; i++){
    small_kernel<<<4,1024>>>(A);
  }
  NDT_ERR
  cudaDeviceSynchronize();
  ndt_cpu_timer_end(start, &time1);
  ndt_timer_print(time1);

  ndt_cpu_timer_start(&start);
  /* run with CUDA Dynamic Parallelism */
  //  run100<<<1,1>>>(A,num);
  cudaDeviceSynchronize();
  ndt_cpu_timer_end(start, &time2);
  ndt_timer_print(time2);
  NDT_ERR
  cudaDeviceReset();
  
  return 0;  
}
