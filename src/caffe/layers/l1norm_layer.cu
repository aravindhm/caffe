// Copyright 2013 Yangqing Jia

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include <thrust/device_vector.h>

#include <eigen3/Eigen/Core>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>

using std::max;

namespace caffe {

template <typename Dtype>
void L1NormLayer<Dtype>::SetUp(
  const vector<Blob<Dtype>*>& bottom, vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Loss Layer takes one blob as input.";
  CHECK_EQ(top->size(), 1) << "Loss Layer takes one output.";
  (*top)[0]->Reshape(bottom[0]->num(), 1, 1, 1); 
  //one output per data point in the mini batch
}

template <typename Dtype>
void L1NormLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype output;
    for(int i = 0; i < num; i++) {
      output = caffe_l1norm(count/num, bottom_data + i*(count/num));
      ((Dtype*)(*top)[0] -> mutable_cpu_data())[i] = output;
    }
}

template <typename Dtype>
void L1NormLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
    int count = bottom[0]->count();
    int num = bottom[0]->num();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype output;
    for(int i = 0; i < num; i++) {
      output = caffe_gpu_l1norm(count/num, bottom_data + i*(count/num));
      ((Dtype*)(*top)[0] -> mutable_cpu_data())[i] = output;
    }
}

template <typename Dtype>
Dtype L1NormLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  if(propagate_down) {
    const Dtype top_diff = *(top[0]->cpu_diff());
    int count = (*bottom)[0]->count();
    int num = (*bottom)[0]->num();
    Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
    memset(bottom_diff, 0, count*sizeof(Dtype));
    const Dtype* bottom_data = (*bottom)[0]->cpu_data();
    for(int i = 0; i < num; i++) {
#pragma omp parallel for
       for(int j = 0; j < (count/num); j++) {
          if(bottom_data[i*(count/num) + j] > 0) {
              bottom_diff[j] += 1;
          }
          else if(bottom_data[i*(count/num) + j] < 0) { 
              bottom_diff[j] += -1;
          }
       }
    }
#pragma omp parallel for
    for(int j = 0; j < (count/num); j++) {
        bottom_diff[j] /= num;
    }
  }
  return Dtype(0);
}

/*
template <typename Dtype>
Dtype L1NormLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
*/

INSTANTIATE_CLASS(L1NormLayer);


}  // namespace caffe
