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
class L1NormLayerTest : public ::testing::Test {
 protected:
  L1NormLayerTest()
      : blob_bottom_data_(new Blob<Dtype>(1, 3, 10, 30)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~L1NormLayerTest() {
    delete blob_bottom_data_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(L1NormLayerTest, Dtypes);

TYPED_TEST(L1NormLayerTest, TestSetUp) {
  LayerParameter layer_param;
  shared_ptr<L1NormLayer<TypeParam> > layer(
  	new L1NormLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
}


TYPED_TEST(L1NormLayerTest, TestCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  shared_ptr<L1NormLayer<TypeParam> > layer(
  	new L1NormLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int n = 0; n < 1; ++n) {
    TypeParam sum = 0;
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 10; ++h) {
        for (int w = 0; w < 30; ++w) {
          sum += fabs(this->blob_bottom_vec_[0]->data_at(n,c,h,w));
        }
      }
    }
    EXPECT_LE(this->blob_top_vec_[0]->data_at(n, 0, 0, 0) - 1e-4, sum);
    EXPECT_GE(this->blob_top_vec_[0]->data_at(n, 0, 0, 0) + 1e-4, sum); 
  }
}

TYPED_TEST(L1NormLayerTest, TestGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  shared_ptr<L1NormLayer<TypeParam> > layer(
  	new L1NormLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  for (int n = 0; n < 1; ++n) {
    TypeParam sum = 0;
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 10; ++h) {
        for (int w = 0; w < 30; ++w) {
          sum += fabs(this->blob_bottom_vec_[0]->data_at(n,c,h,w));
        }
      }
    }
    EXPECT_LE(this->blob_top_vec_[0]->data_at(n, 0, 0, 0) - 1e-4, sum);
    EXPECT_GE(this->blob_top_vec_[0]->data_at(n, 0, 0, 0) + 1e-4, sum); 
  }
}
/*
TYPED_TEST(L1NormLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  L1NormLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(L1NormLayerTest, TestGradientGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  L1NormLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
*/
}
