// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"
#include <thrust/device_vector.h>
#include <thrust/transform.h>

#include <iostream>

namespace caffe {

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y) {
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y) {
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  for(int i = threadIdx.x + blockIdx.x * blockDim.x; 
        i < n;
        i += blockDim.x + gridDim.x)
  {
     y[i] = a[i] - b[i];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a,
    const float* b, float* y) {
  int deviceid;
  cudaGetDevice(&deviceid);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceid);
  sub_kernel<float><<<numSMs, CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  /*thrust::device_ptr<const float> dev_ptr_a(a);
  thrust::device_ptr<const float> dev_ptr_b(b);
  thrust::device_ptr<float> dev_ptr_y(y);
  thrust::transform(dev_ptr_a, dev_ptr_a + N, dev_ptr_b, dev_ptr_y, thrust::minus<float>());
  */
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a,
    const double* b, double* y) {
  int deviceid;
  cudaGetDevice(&deviceid);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceid);
  sub_kernel<double><<<numSMs, CAFFE_CUDA_NUM_THREADS>>>(
      N, a, b, y);
  /*thrust::device_ptr<const double> dev_ptr_a(a);
  thrust::device_ptr<const double> dev_ptr_b(b);
  thrust::device_ptr<double> dev_ptr_y(y);
  thrust::transform(dev_ptr_a, dev_ptr_a + N, dev_ptr_b, dev_ptr_y, thrust::minus<double>());
  */
}

//The below unary operator will be used by thrust to compute the L1 norm.
template<typename Dtype>
struct abs
{
   __host__ __device__
       Dtype operator()(const Dtype& x) const {
          if(x >= 0.00000) return x;
          else return -x;
       }
};

template <>
float caffe_gpu_l1norm<float>(const int N, const float* a) {
   thrust::device_ptr<const float> dev_ptr_a(a);
   return thrust::transform_reduce(dev_ptr_a, dev_ptr_a + N, 
           abs<float>(), (float)0.0, thrust::plus<float>());
}

template <>
double caffe_gpu_l1norm<double>(const int N, const double* a) {
   thrust::device_ptr<const double> dev_ptr_a(a);
   return thrust::transform_reduce(dev_ptr_a, dev_ptr_a + N, 
           abs<double>(), (double)0.0, thrust::plus<double>());
}

}  // namespace caffe
