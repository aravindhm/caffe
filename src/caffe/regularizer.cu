// Copyright 2014 kloudkl@github

#include <cmath> // for std::abs

#include "caffe/proto/caffe.pb.h"
#include "caffe/regularizer.hpp"
#include "caffe/util/math_functions.hpp" // for caffe_gpu_asum

namespace caffe {

template<typename Dtype>
Dtype Regularizer<Dtype>::Regularize(Blob<Dtype>* bottom) {
  Dtype penalty = 0;
  if (Caffe::mode() == Caffe::CPU) {
    penalty = Regularize_cpu(bottom);
  } else if (Caffe::mode() == Caffe::GPU) {
    penalty = Regularize_gpu(bottom);
  } else {
    LOG(FATAL) << "Unknown mode: " << Caffe::mode();
  }
  return penalty;
}

template<typename Dtype>
Dtype L1Regularizer<Dtype>::Regularize_cpu(Blob<Dtype>* bottom) {
  if (this->coeff_ == 0) {
    return Dtype(0);
  }
  const Dtype* data = bottom->cpu_data();
  Dtype* diff = bottom->mutable_cpu_diff();
  int count = bottom->count();
  Dtype penalty = 0;
  for (int c = 0; c < count; ++c) {
    diff[c] += this->coeff_ * sign < Dtype > (data[c]);
    penalty += std::abs(data[c]);
  }
  return this->coeff_ * penalty;
}

template <typename Dtype>
__device__ inline int gpu_sign(const Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

template __device__ int gpu_sign<float>(const float val);
template __device__ int gpu_sign<double>(const double val);

template <typename Dtype>
__global__ void ScaleSign(const int n, const Dtype coeff, const Dtype* data, Dtype* diff) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    diff[index] += coeff * gpu_sign<Dtype>(data[index]);
  }
}

template<typename Dtype>
Dtype L1Regularizer<Dtype>::Regularize_gpu(Blob<Dtype>* bottom) {
  if (this->coeff_ == 0) {
    return Dtype(0);
  }
  const Dtype* data = bottom->gpu_data();
  Dtype* diff = bottom->mutable_gpu_diff();
  int count = bottom->count();
  ScaleSign<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, this->coeff_, data, diff);
  CUDA_POST_KERNEL_CHECK;
  Dtype penalty = 0;
  caffe_gpu_asum < Dtype > (count, data, &penalty);
  LOG(INFO) << this->coeff_ * penalty;
  return this->coeff_ * penalty;
}

template<typename Dtype>
Dtype L2Regularizer<Dtype>::Regularize_cpu(Blob<Dtype>* bottom) {
  if (this->coeff_ == 0) {
    return Dtype(0);
  }
  const Dtype* data = bottom->cpu_data();
  Dtype* diff = bottom->mutable_cpu_diff();
  int count = bottom->count();
  caffe_axpy < Dtype > (count, this->coeff_ * 2., data, diff);
  Dtype penalty = caffe_cpu_dot < Dtype > (count, data, data);
  return this->coeff_ * penalty;
}

template<typename Dtype>
Dtype L2Regularizer<Dtype>::Regularize_gpu(Blob<Dtype>* bottom) {
  if (this->coeff_ == 0) {
    return Dtype(0);
  }
  const Dtype* data = bottom->gpu_data();
  Dtype* diff = bottom->mutable_gpu_diff();
  int count = bottom->count();
  caffe_gpu_axpy < Dtype > (count, this->coeff_ * 2., data, diff);
  Dtype penalty = 0;
  caffe_gpu_dot < Dtype > (count, data, data, &penalty);
  return this->coeff_ * penalty;
}

template<typename Dtype>
Dtype MaxNormRegularizer<Dtype>::Regularize_cpu(Blob<Dtype>* bottom) {
  if (this->coeff_ == 0) {
    return Dtype(0);
  }
  const Dtype* data = bottom->cpu_data();
  Dtype* diff = bottom->mutable_cpu_diff();
  int count = bottom->count();
  Dtype penalty = 0;
  // TODO: Implement MaxNormRegularizer::Regularize_cpu
  return this->coeff_ * penalty;
}

template<typename Dtype>
Dtype MaxNormRegularizer<Dtype>::Regularize_gpu(Blob<Dtype>* bottom) {
  if (this->coeff_ == 0) {
    return Dtype(0);
  }
  const Dtype* data = bottom->cpu_data();
  Dtype* diff = bottom->mutable_cpu_diff();
  int count = bottom->count();
  Dtype penalty = 0;
  // TODO: Implement MaxNormRegularizer::Regularize_cpu
  return this->coeff_ * penalty;
}

template<typename Dtype>
Regularizer<Dtype>* GetRegularizer(const LayerParameter& param) {
  const LayerParameter_RegularizerType type = param.regularizer();
  if (type == REG_TYPE(L1)) {
    return new L1Regularizer<Dtype>(param);
  } else if (type == REG_TYPE(L2)) {
    return new L2Regularizer<Dtype>(param);
  } else if (type == REG_TYPE(MAX_NORM)) {
    return new MaxNormRegularizer<Dtype>(param);
  } else {
    LOG(FATAL) << "Unknown regularizer type: " << type;
  }
  // just to suppress old compiler warnings.
  return (Regularizer<Dtype>*) (NULL);
}

template Regularizer<float>* GetRegularizer<float>(const LayerParameter& param);
template Regularizer<double>* GetRegularizer<double>(
    const LayerParameter& param);

INSTANTIATE_CLASS (Regularizer);
INSTANTIATE_CLASS (L1Regularizer);
INSTANTIATE_CLASS (L2Regularizer);
INSTANTIATE_CLASS (MaxNormRegularizer);

}  // namespace caffe
