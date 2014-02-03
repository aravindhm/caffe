// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <vector>
#include <thrust/device_vector.h>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

using std::max;

namespace caffe {

template <typename Dtype>
void EuclideanLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 2) << "Loss Layer takes two blobs as input.";
  CHECK_EQ(top->size(), 1) << "Loss Layer takes one output.";
  CHECK_EQ(bottom[0]->num(), bottom[1]->num())
      << "The data1 and data2 should have the same number.";
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  difference_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      bottom[0]->height(), bottom[0]->width());
  (*top)[0]->Reshape(1, 1, 1, 1);
}

template <typename Dtype>
void EuclideanLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    caffe_sub(count, bottom[0]->cpu_data(), bottom[1]->cpu_data(),
        difference_.mutable_cpu_data());
    Dtype loss = caffe_cpu_dot(
        count, difference_.cpu_data(), difference_.cpu_data()) / num;
    (*top)[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    caffe_gpu_sub(count, bottom[0]->gpu_data(), bottom[1]->gpu_data(),
        difference_.mutable_gpu_data());
    Dtype loss;
    caffe_gpu_dot(count, 
        difference_.gpu_data(), difference_.gpu_data(), &loss);
    (*top)[0]->mutable_cpu_data()[0] = loss/num;
}

template <typename Dtype>
Dtype EuclideanLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if(propagate_down) {
    const Dtype top_diff = *(top[0]->cpu_diff());
    int count = (*bottom)[0]->count();
    int num = (*bottom)[0]->num();
    caffe_sub(count, (*bottom)[0]->cpu_data(), (*bottom)[1]->cpu_data(),
        difference_.mutable_cpu_data());
    //This is not a layer layer type. Its computing the error and sending it further down the network. 
    //Dtype loss = caffe_cpu_dot(
    //    count, difference_.cpu_data(), difference_.cpu_data()) / num / Dtype(2);
    // Compute the gradient
    caffe_axpby(count, top_diff / num, difference_.cpu_data(), Dtype(0),
        (*bottom)[0]->mutable_cpu_diff());
  }
  return Dtype(0);
}

template <typename Dtype>
Dtype EuclideanLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if(propagate_down) {
    const Dtype top_diff = *(top[0]->cpu_diff());
    int count = (*bottom)[0]->count();
    int num = (*bottom)[0]->num();
    caffe_gpu_sub(count, (*bottom)[0]->gpu_data(), (*bottom)[1]->gpu_data(),
        difference_.mutable_gpu_data());
    //This is not a layer layer type. Its computing the error and sending it further down the network. 
    //Dtype loss;
    //caffe_gpu_dot(
    //    count, difference_.gpu_data(), difference_.gpu_data(), &loss);
    //loss = loss / num / Dtype(2);
    // Compute the gradient
    caffe_gpu_axpby(count, top_diff / num, difference_.gpu_data(), Dtype(0),
        (*bottom)[0]->mutable_gpu_diff());
  }
  return Dtype(0);
}


INSTANTIATE_CLASS(EuclideanLayer);


}  // namespace caffe
