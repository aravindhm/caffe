// Copyright 2014 Aravindh Mahendran

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>

namespace caffe {

template <typename Dtype>
void ComputeCovarianceLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "ComputeCovariance takes a single blob as input";
  CHECK_EQ(top->size(), 1) << "I output a single covarinace only. Use multiple of me for several different inputs and outputs.";
  K_ = bottom[0]->count()/bottom[0]->num(); // the size of each image is the number of rows in my x matrix
  M_ = bottom[0]->num(); // the number of elements is the width of my x matrix
  // in  other words x is #pixels x #images matrix
  // the covariance matrix is then #pixels x #pixels obtained as 1/K_ * x * x'
  //std::cout << "Reshape to " << 1 << " " << 1 << " " << K_ << " " << K_ << std::endl;
  (*top)[0] -> Reshape(1, 1, K_, K_);
  //std::cout << "Reshape done" << std::endl;
  total_data_ = 0;
}

template <typename Dtype>
void ComputeCovarianceLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_mutable_data = bottom[0]->mutable_cpu_data();

  // first do the mean subtraction
  // Use a multiplication with ones to compute the mean - it does repmat effectively
  Blob<Dtype> ones(1, 1, K_, K_);
  Dtype* ones_mutable_data = ones.mutable_cpu_data();
  const Dtype* ones_data = ones.cpu_data();
  for(int i = 0; i < ones.count(); i++) {
    ones_mutable_data[i] = 1.;
  }
 
  Blob<Dtype> mean(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  Dtype* mean_mutable_data = mean.mutable_cpu_data();
  const Dtype* mean_data = mean.cpu_data();

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
       M_, K_, K_, (Dtype)1./K_, 
       bottom_data, ones_data, (Dtype)0., mean_mutable_data);  

  // Now subtract the mean
  caffe_sub<Dtype>(bottom[0]->count(), bottom_data, mean_data, bottom_mutable_data);

  // Now that bottom_data is mean subtracted. Compute the covariance.
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 
       K_, K_, M_, (Dtype)1./(total_data_ + M_),
       bottom_data, bottom_data,
       ((Dtype)total_data_)/(total_data_ + M_),
       top_data);
  total_data_ = total_data_ + M_;
}

template <typename Dtype>
void ComputeCovarianceLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  //std::cout << "Getting the pointers for bottom_data" << std::endl;
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_mutable_data = bottom[0]->mutable_gpu_data();

  // first do the mean subtraction
  // Use a multiplication with ones to compute the mean - it does repmat effectively
  //std::cout << "Creating the ones array" << std::endl;
  Blob<Dtype> ones(1, 1, K_, K_);
  Dtype* ones_mutable_data = ones.mutable_gpu_data();
  const Dtype* ones_data = ones.gpu_data();
  caffe_gpu_valset(ones.count(), ones_mutable_data, (Dtype)1.);
 
  //std::cout << "Computing the mean" << std::endl;
  Blob<Dtype> mean(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  Dtype* mean_mutable_data = mean.mutable_gpu_data();
  const Dtype* mean_data = mean.gpu_data();

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
       M_, K_, K_, (Dtype)1./K_, 
       bottom_data, ones_data, (Dtype)0., mean_mutable_data);  

  // Now subtract the mean
  //std::cout << "Subtracting the mean" << std::endl;
  caffe_gpu_sub<Dtype>(bottom[0]->count(), bottom_data, mean_data, bottom_mutable_data);

  // Now that bottom_data is mean subtracted. Compute the covariance.
  //std::cout << "Updating the covariance" << std::endl;
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  caffe_gpu_gemm<Dtype>(CblasTrans, CblasNoTrans, 
       K_, K_, M_, (Dtype)1./(total_data_ + M_),
       bottom_data, bottom_data,
       ((Dtype)total_data_)/(total_data_ + M_),
       top_data);
  total_data_ = total_data_ + M_;
/*
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* bottom_mutable_data = bottom[0]->mutable_gpu_data();

  // first do the mean subtraction
  // Use a multiplication with ones to compute the mean - it does repmat effectively
  Blob<Dtype> ones(1, 1, M_, M_);
  Dtype* ones_mutable_data = ones.mutable_gpu_data();
  const Dtype* ones_data = ones.gpu_data();
  caffe_gpu_valset(ones.count(), ones_mutable_data, (Dtype)1.);
  Blob<Dtype> mean(bottom[0]->num(), bottom[0]->channels(), bottom[0]->height(), bottom[0]->width());
  Dtype* mean_mutable_data = mean.mutable_gpu_data();
  const Dtype* mean_data = mean.gpu_data();

  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans,
       M_, K_, M_, (Dtype)1./M_, 
       ones_data, bottom_data, (Dtype)0., mean_mutable_data);  

  // Now subtract the mean
  caffe_gpu_sub<Dtype>(bottom[0]->count(), bottom_data, mean_data, bottom_mutable_data);

  // Now that bottom_data is mean subtracted. Compute the covariance.
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 
       M_, N_, K_, (Dtype)1./(total_data_ + K_),
       bottom_data, bottom_data,
       ((Dtype)total_data_)/(total_data_ + K_),
       top_data);
  total_data_ = total_data_ + K_;
*/
}

INSTANTIATE_CLASS(ComputeCovarianceLayer);

}
