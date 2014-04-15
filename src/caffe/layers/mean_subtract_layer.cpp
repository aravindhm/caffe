// Copyright 2014 Aravindh Mahendran


//#include <mkl.h>
#include <cublas_v2.h>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void MeanSubtractLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
 CHECK_EQ(bottom.size(), 1) << "MeanSubtract Layer takes a single blob as input.";
 CHECK_EQ(top->size(), 1) << "MeanSubtract Layer takes a single blob as output.";
}

template <typename Dtype>
void MeanSubtractLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_data_mutable = bottom[0]->mutable_cpu_data();
  Dtype* top_data_mutable = (*top)[0]->mutable_cpu_data();
  int dim = bottom[0]->count()/bottom[0]->num();
  // First subtract the mean
  for(int i = 0; i < bottom[0]->num(); i++) {
    Dtype mean = caffe_cpu_mean(dim, bottom_data + i*dim);
    caffe_sub(dim, bottom_data+i*dim, mean, top_data_mutable+i*dim);
  }
}

template <typename Dtype>
void MeanSubtractLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_data_mutable = bottom[0]->mutable_gpu_data();
  Dtype* top_data_mutable = (*top)[0]->mutable_gpu_data();
  int dim = bottom[0]->count()/bottom[0]->num();
  // First subtract the mean
  for(int i = 0; i < bottom[0]->num(); i++) {
    Dtype mean = caffe_gpu_mean(dim, bottom_data + i*dim);
    caffe_gpu_sub(dim, bottom_data+i*dim, mean, top_data_mutable+i*dim);
  }
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype MeanSubtractLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype MeanSubtractLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(MeanSubtractLayer);

} // namespace caffe
