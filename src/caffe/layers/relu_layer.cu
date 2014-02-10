// Copyright 2013 Yangqing Jia

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include <algorithm>

#include <iostream>

#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

using std::max;

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const int count = bottom[0]->count();
  for (int i = 0; i < count; ++i) {
    top_data[i] = max(bottom_data[i], Dtype(0));
  }
}

template <typename Dtype>
Dtype ReLULayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    const int count = (*bottom)[0]->count();
    for (int i = 0; i < count; ++i) {
      bottom_diff[i] = top_diff[i] * (bottom_data[i] > 0);
    }
  }
  return Dtype(0);
}

template <typename Dtype>
__global__ void ReLUForward(const int n, const Dtype* in, Dtype* out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out[index] = in[index] > 0 ? in[index] : 0;
  }
}

template <typename Dtype>
struct relu_functor
{
    relu_functor() {}
    __host__ __device__
        Dtype operator()(const Dtype& x) const { 
            if(x > 0) return x;
            else return 0;
        }
};


template <typename Dtype>
void ReLULayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = bottom[0]->count();
  thrust::device_ptr<const Dtype> dev_ptr_bottom_data(bottom_data);
  thrust::device_ptr<Dtype> dev_ptr_top_data(top_data);
  thrust::transform(dev_ptr_bottom_data, dev_ptr_bottom_data+count, dev_ptr_top_data, 
       relu_functor<Dtype>());
/*  ReLUForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data);
  CUDA_POST_KERNEL_CHECK; */
  // << " count: " << count << " bottom_data: "
  //     << (unsigned long)bottom_data << " top_data: " << (unsigned long)top_data
  //     << " blocks: " << CAFFE_GET_BLOCKS(count)
  //     << " threads: " << CAFFE_CUDA_NUM_THREADS;
}

template <typename Dtype>
__global__ void ReLUBackward(const int n, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    out_diff[index] = in_diff[index] * (in_data[index] > 0);
  }
}

template <typename Dtype>
Dtype ReLULayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* bottom_data = (*bottom)[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = (*bottom)[0]->count();
    ReLUBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_data, bottom_diff);
    CUDA_POST_KERNEL_CHECK;
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(ReLULayer);


}  // namespace caffe
