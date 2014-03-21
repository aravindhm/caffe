// Copyright 2013 Yangqing Jia

#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

#include <iostream>

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class ComputeCovarianceLayerTest : public ::testing::Test {
 protected:
  ComputeCovarianceLayerTest()
      : blob_bottom_(new Blob<Dtype>(1, 1, 2, 2)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    filler_param.set_mean(0.0);
    filler_param.set_std(1.0);
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~ComputeCovarianceLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(ComputeCovarianceLayerTest, Dtypes);

TYPED_TEST(ComputeCovarianceLayerTest, TestSetUp) {
  LayerParameter layer_param;
  ComputeCovarianceLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 4);
  EXPECT_EQ(this->blob_top_->channels(), 4);
}

TYPED_TEST(ComputeCovarianceLayerTest, TestCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ComputeCovarianceLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  
  FillerParameter filler_param;
  filler_param.set_mean(0.0);
  filler_param.set_std(1.0);
  GaussianFiller<TypeParam> filler(filler_param);
  TypeParam* mutable_data = this->blob_bottom_->mutable_cpu_data();
  for(int i = 0; i < 10000; i++) {
    // simulate gaussian random noise for the data 
    // and compute covariane. 
    // over a 10000 iterations the output should converge to the Identity as all the parameters are filled IID
    //filler.Fill(this->blob_bottom_);
    mutable_data[0] = rand()/((double)RAND_MAX+1);
    mutable_data[1] = rand()/((double)RAND_MAX+1);
    mutable_data[2] = rand()/((double)RAND_MAX+1);
    mutable_data[3] = rand()/((double)RAND_MAX+1);
    printf("%f\n", this->blob_bottom_->data_at(0,0,0,0));
    layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int width = this->blob_top_->width();
  const int height = this->blob_top_->height();
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
        printf("%f ", this->blob_top_->data_at(0,0,i,j));
    }
    printf("\n");
  }
  }
}

TYPED_TEST(ComputeCovarianceLayerTest, TestGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  ComputeCovarianceLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  
  FillerParameter filler_param;
  filler_param.set_mean(0.0);
  filler_param.set_std(1.0);
  GaussianFiller<TypeParam> filler(filler_param);
  for(int i = 0; i < 10000; i++) {
    // simulate gaussian random noise for the data 
    // and compute covariane. 
    // over a 10000 iterations the output should converge to the Identity as all the parameters are filled IID
    filler.Fill(this->blob_bottom_);
    printf("%f\n", this->blob_bottom_->data_at(0,0,0,0));
    layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int width = this->blob_top_->width();
  const int height = this->blob_top_->height();
  for (int i = 0; i < height; ++i) {
    for (int j = 0; j < width; ++j) {
        printf("%f ", this->blob_top_->data_at(0,0,i,j));
    }
    printf("\n");
  }
  }
}
}
