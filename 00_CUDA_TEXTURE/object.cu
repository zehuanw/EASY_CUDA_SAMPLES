/********************************************************************************
 * TEX Object API 
 *
 * TODO:
 * Test the behavior of memory cache of cuArray and 2D pitched memory tex.
 * Test the behavior of float
 * I suspect some other unit can be used in analysis.
 *******************************************************************************/
#include <stdio.h>

#define DIM 2
//#define DATA_TYPE unsigned int //please note that if change this you many also need change cudaCreateChannelDesc
//#define DATA_TYPE float //please note that if change this you many also need change cudaCreateChannelDesc
#define DATA_TYPE unsigned char //please note that if change this you many also need change cudaCreateChannelDesc
#define DIM_1 16
#define DIM_2 16
#define DIM_3 16
#define CUDA_ARRAY


#if DIM == 1
#elif DIM == 2
__global__ void kernel(cudaTextureObject_t texObj){
//  DATA_TYPE a = tex2D<DATA_TYPE>(texObj,2,1);//note: require a <DATA_TYPE> in object api version.
  float a = tex2D<float>(texObj,1,0);//note: require a <DATA_TYPE> in object api version.
  //printf("%d\n",(int)a);
  printf("%f\n",a);
  return;
}
#elif DIM == 3
#endif 


int main(){
  DATA_TYPE* hostPtr = (DATA_TYPE*)malloc(DIM_1*DIM_2*DIM_3*sizeof(DATA_TYPE));
  for(int i=0;i<DIM_1*DIM_2*DIM_3;i++)
  {
    hostPtr[i] = i;
  }
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(DATA_TYPE)*8,0,0,0,cudaChannelFormatKindUnsigned);
  //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(DATA_TYPE)*8,0,0,0,cudaChannelFormatKindFloat);
  
  struct cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeBorder;
  texDesc.addressMode[1] = cudaAddressModeBorder;
  //texDesc.filterMode = cudaFilterModePoint;
  texDesc.filterMode = cudaFilterModeLinear; //only support when cudaCreateChannelDesc ==  cudaChannelFormatKindFloat
  //texDesc.readMode = cudaReadModeElementType;
  texDesc.readMode = cudaReadModeNormalizedFloat;
  texDesc.normalizedCoords = 0;
  //texDesc.normalizedCoords = 1;
  cudaTextureObject_t texObj = 0;
  
  struct cudaResourceDesc resDesc; //refer to the def of cudaCreateTextureObject in cuda_runtime_api.h
  memset(&resDesc, 0, sizeof(resDesc));  

  
#ifndef CUDA_ARRAY
  DATA_TYPE* devPtr;
#if DIM == 1
#elif DIM == 2
  size_t pitch;
  cudaMallocPitch(&devPtr,&pitch,DIM_1*sizeof(DATA_TYPE),DIM_2);
  cudaMemcpy2D(devPtr,pitch,hostPtr,DIM_1*sizeof(DATA_TYPE),DIM_1*sizeof(DATA_TYPE),DIM_2,cudaMemcpyHostToDevice);

  resDesc.resType = cudaResourceTypePitch2D;
  resDesc.res.pitch2D.devPtr = devPtr;
  resDesc.res.pitch2D.desc = channelDesc;
  resDesc.res.pitch2D.width = DIM_1; //should be in element not in byte
  resDesc.res.pitch2D.height = DIM_2;
  resDesc.res.pitch2D.pitchInBytes = pitch;
  cudaCreateTextureObject(&texObj,&resDesc,&texDesc,NULL);
#elif DIM == 3
#endif//#if DIM == 1
  
#else
#if DIM == 1
#elif DIM == 2
  cudaArray* cuArray_2d;
  cudaExtent extent_2d = {DIM_1,DIM_2,0};
  cudaMalloc3DArray(&cuArray_2d, &channelDesc,extent_2d ,cudaArrayDefault);//this function is able to alloc 1/3D array there are some interesting choice for the 4th parameter. Note zero in z of extent_2d.
  cudaMemcpy3DParms cpy3DParms = {0}; //should be init to zero before use

  cpy3DParms.srcPtr = make_cudaPitchedPtr(hostPtr,DIM_1*sizeof(DATA_TYPE),DIM_1,DIM_2);
  cpy3DParms.dstArray = cuArray_2d;
  cpy3DParms.extent = make_cudaExtent(DIM_1,DIM_2,1); //If no CUDA array is participating in the copy then the extents are defined in elements of unsigned char.
  cpy3DParms.kind = cudaMemcpyHostToDevice;
  cudaMemcpy3D(&cpy3DParms);

  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray_2d;
  cudaCreateTextureObject(&texObj,&resDesc,&texDesc,NULL);
#elif DIM == 3
#endif//#if DIM == 1
#endif//CUDA_ARRAY

  kernel<<<1,1>>>(texObj);
  cudaDeviceSynchronize();
  printf("%s\n",cudaGetErrorString(cudaGetLastError()));

  return 0;
}
