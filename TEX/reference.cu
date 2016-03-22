/**************************************************************
 * TEX Reference API
 *
 * TODO:
 * Add layered Texture support to TYPE
 *************************************************************/

#define DIM 2
#define DATA_TYPE unsigned int //please note that if change this you many also need change cudaCreateChannelDesc
#define READ_MODE cudaReadModeElementType
//#define READ_MODE cudaReadModeNormalizedFloat
#define HIGH_LEVEL


#define DIM_1 16
#define DIM_2 16
#define DIM_3 16
#define CUDA_ARRAY



#if DIM == 1
#define TYPE cudaTextureType1D
#elif DIM == 2
#define TYPE cudaTextureType2D
#elif DIM == 3
#define TYPE cudaTextureType3D
#endif 

#include <stdio.h>
texture<DATA_TYPE, TYPE, READ_MODE> texRef; //specified in compiling time

#if DIM == 1
#elif DIM == 2
__global__ void kernel(){
  DATA_TYPE a = tex2D(texRef,18,1);
  printf("%d\n",(int)a);
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

  textureReference* texRefPtr = NULL;
  const textureReference* constTexRefPtr = NULL;

  cudaGetTextureReference(&constTexRefPtr, &texRef);//don't use "texRef" here, that def is deprecate.
  texRefPtr = const_cast<textureReference*>(constTexRefPtr);
  
  {
  // texRefPtr->addressMode[0] = cudaAddressModeClamp;
  // texRefPtr->addressMode[1] = cudaAddressModeClamp;
  texRefPtr->addressMode[0] = cudaAddressModeBorder;
  texRefPtr->addressMode[1] = cudaAddressModeBorder;
  texRefPtr->filterMode = cudaFilterModePoint;
  //texRefPtr->filterMode = cudaFilterModeLinear;
  texRefPtr->normalized = 0; //whether texture coordinates are normalized or not
  //There are others can be modified please see in texture_types.h
  }  

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<DATA_TYPE>();
  channelDesc.x = sizeof(DATA_TYPE)*8; //channel width of first channel (in bit);
  channelDesc.y = 0;
  channelDesc.z = 0;
  channelDesc.w = 0;
  channelDesc.f = cudaChannelFormatKindUnsigned;

  //cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(sizeof(DATA_TYPE)*8,0,0,0,cudaChannelFormatKindUnsigned); //an equivalence of the previous one. 

#ifndef CUDA_ARRAY
  DATA_TYPE* devPtr;
#if DIM == 1
#elif DIM == 2
  size_t pitch;
  cudaMallocPitch(&devPtr,&pitch,DIM_1*sizeof(DATA_TYPE),DIM_2);
  cudaMemcpy2D(devPtr,pitch,hostPtr,DIM_1*sizeof(DATA_TYPE),DIM_1*sizeof(DATA_TYPE),DIM_2,cudaMemcpyHostToDevice);
#elif DIM == 3
#endif //#if DIM == 1

  size_t offset;
#ifdef HIGH_LEVEL
  cudaBindTexture2D(&offset, &texRef, devPtr, &channelDesc, DIM_1,DIM_2,pitch);
#else 
  cudaBindTexture2D(&offset, texRefPtr, devPtr, &channelDesc, DIM_1,DIM_2,pitch);
#endif //HIGH_LEVEL

#else //#ifndef CUDA_ARRAY
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
#elif DIM == 3
#endif//#if DIM == 1
  

#ifdef HIGH_LEVEL
  cudaBindTextureToArray(texRef,cuArray_2d);
#else 
  cudaBindTextureToArray(texRef,cuArray_2d,&channelDesc);
#endif //HIGH_LEVEL
#endif //#ifndef CUDA_ARRAY


  kernel<<<1,1>>>();
  cudaDeviceSynchronize();
  printf("%s\n",cudaGetErrorString(cudaGetLastError()));
  return 0;
}
