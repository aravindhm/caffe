// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstdlib>
#include <cstring>

#include "caffe/common.hpp"
#include "caffe/util/math_functions.hpp"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>

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

/* grid stride kernel */
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
}


/* grid stride kernel */
template <typename Dtype>
__global__ void sub_scalar_kernel(const int n, const Dtype* x, const Dtype val, 
    Dtype* y) {
  for(int i = threadIdx.x + blockIdx.x * blockDim.x; 
        i < n;
        i += blockDim.x + gridDim.x)
  {
     y[i] = x[i] - val;
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* x, const float val, float* y) {
  int deviceid;
  cudaGetDevice(&deviceid);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceid);
  sub_scalar_kernel<float><<<numSMs, CAFFE_CUDA_NUM_THREADS>>>(
      N, x, val, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* x, const double val, double* y) {
  int deviceid;
  cudaGetDevice(&deviceid);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceid);
  sub_scalar_kernel<double><<<numSMs, CAFFE_CUDA_NUM_THREADS>>>(
      N, x, val, y);
}


/* grid stride kernel */
template <typename Dtype>
__global__ void valset_kernel(const int n, Dtype* y, Dtype val) {
  for(int i = threadIdx.x + blockIdx.x * blockDim.x; 
        i < n;
        i += blockDim.x + gridDim.x)
  {
     y[i] = val;
  }
}

template <>
void caffe_gpu_valset<float>(const int N, float* y, float val) {
  int deviceid;
  cudaGetDevice(&deviceid);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceid);
  valset_kernel<float><<<numSMs, CAFFE_CUDA_NUM_THREADS>>>(
      N, y, val);
}

template <>
void caffe_gpu_valset<double>(const int N, double* y, double val) {
  int deviceid;
  cudaGetDevice(&deviceid);
  int numSMs;
  cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, deviceid);
  valset_kernel<double><<<numSMs, CAFFE_CUDA_NUM_THREADS>>>(N, y, val);
}

template <>
float caffe_gpu_mean<float>(const int N, const float* x) {
    thrust::device_ptr<const float> dev_ptr(x);
    thrust::device_vector<float> d_x(dev_ptr, dev_ptr + N);
    // compute norm
    float sum = thrust::reduce(d_x.begin(), d_x.end());
    return sum/N;
}

template <>
double caffe_gpu_mean<double>(const int N, const double* x) {
    thrust::device_ptr<const double> dev_ptr(x);
    thrust::device_vector<double> d_x(dev_ptr, dev_ptr + N);

    // compute norm
    double sum = thrust::reduce(d_x.begin(), d_x.end());
    return sum/N;
}

}  // namespace caffe
