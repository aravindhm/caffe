// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class EuclideanLayerTest : public ::testing::Test {
 protected:
  EuclideanLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
  }
  virtual ~EuclideanLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* blob_top_data_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(EuclideanLayerTest, Dtypes);

TYPED_TEST(EuclideanLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  EuclideanLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0, -1, -1);
}

TYPED_TEST(EuclideanLayerTest, TestGradientGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  EuclideanLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0, -1, -1);
}

TYPED_TEST(EuclideanLayerTest, TestCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  EuclideanLayer<TypeParam> layer(layer_param);
  this->blob_top_data_ = new Blob<TypeParam>(1,1,1,1);
  //I'm doing this here because I don't know how to change
  // the gradient checks when top is non empty. If I do these up in
  // the constructor the call to the gradient checker segfaults.
  this->blob_top_vec_.push_back(this->blob_top_data_);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_vec_[0]->num(), 1);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(), 1);
  EXPECT_EQ(this->blob_top_vec_[0]->height(), 1);
  EXPECT_EQ(this->blob_top_vec_[0]->width(), 1);
  TypeParam sum = 0;
  for (int n = 0; n < 10; ++n) {
    for (int c = 0; c < 5; ++c) {
      for (int h = 0; h < 1; ++h) {
        for (int w = 0; w < 1; ++w) {
          sum += pow((this->blob_bottom_vec_[0]->data_at(n,c,h,w) - 
                 this->blob_bottom_vec_[1]->data_at(n,c,h,w)), 2);
        }
      }
    }
  }
  sum = sum / 10.0;
  EXPECT_LE(this->blob_top_vec_[0]->data_at(0, 0, 0, 0) - 1e-4, sum);
  EXPECT_GE(this->blob_top_vec_[0]->data_at(0, 0, 0, 0) + 1e-4, sum); 
}

TYPED_TEST(EuclideanLayerTest, TestGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  EuclideanLayer<TypeParam> layer(layer_param);
  this->blob_top_data_ = new Blob<TypeParam>(1,1,1,1); 
  //I'm doing this here because I don't know how to change
  // the gradient checks when top is non empty. If I do these up in
  // the constructor the call to the gradient checker segfaults.
  this->blob_top_vec_.push_back(this->blob_top_data_);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_vec_[0]->num(), 1);
  EXPECT_EQ(this->blob_top_vec_[0]->channels(), 1);
  EXPECT_EQ(this->blob_top_vec_[0]->height(), 1);
  EXPECT_EQ(this->blob_top_vec_[0]->width(), 1);
  TypeParam sum = 0;
  for (int n = 0; n < 10; ++n) {
    for (int c = 0; c < 5; ++c) {
      for (int h = 0; h < 1; ++h) {
        for (int w = 0; w < 1; ++w) {
          sum += pow((this->blob_bottom_vec_[0]->data_at(n,c,h,w) - 
                 this->blob_bottom_vec_[1]->data_at(n,c,h,w)), 2);
        }
      }
    }
  }
  sum = sum / 10.0;
  EXPECT_LE(this->blob_top_vec_[0]->data_at(0, 0, 0, 0) - 1e-4, sum);
  EXPECT_GE(this->blob_top_vec_[0]->data_at(0, 0, 0, 0) + 1e-4, sum);
}
}
