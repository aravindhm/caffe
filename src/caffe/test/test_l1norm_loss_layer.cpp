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
class L1NormLossLayerTest : public ::testing::Test {
 protected:
  L1NormLossLayerTest()
      : blob_bottom_(new Blob<Dtype>(10, 5, 1, 10)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~L1NormLossLayerTest() {
    delete blob_bottom_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(L1NormLossLayerTest, Dtypes);

TYPED_TEST(L1NormLossLayerTest, TestSetUp) {
  LayerParameter layer_param;
  shared_ptr<L1NormLossLayer<TypeParam> > layer(
  	new L1NormLossLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 1);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
}


TYPED_TEST(L1NormLossLayerTest, TestCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  shared_ptr<L1NormLossLayer<TypeParam> > layer(
  	new L1NormLossLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  TypeParam sum = 0;
  for (int n = 0; n < 10; ++n) {
    for (int c = 0; c < 5; ++c) {
      for (int h = 0; h < 1; ++h) {
        for (int w = 0; w < 10; ++w) {
          sum += fabs(this->blob_bottom_vec_[0]->data_at(n,c,h,w));
        }
      }
    }
  }
  sum = sum /10;
  EXPECT_LE(this->blob_top_vec_[0]->data_at(0, 0, 0, 0) - 1e-4, sum);
  EXPECT_GE(this->blob_top_vec_[0]->data_at(0, 0, 0, 0) + 1e-4, sum); 
}

TYPED_TEST(L1NormLossLayerTest, TestGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  shared_ptr<L1NormLossLayer<TypeParam> > layer(
  	new L1NormLossLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  TypeParam sum = 0;
  for (int n = 0; n < 10; ++n) {
    for (int c = 0; c < 5; ++c) {
      for (int h = 0; h < 1; ++h) {
        for (int w = 0; w < 10; ++w) {
          sum += fabs(this->blob_bottom_vec_[0]->data_at(n,c,h,w));
        }
      }
    }
  }
  sum = sum / 10;
  EXPECT_LE(this->blob_top_vec_[0]->data_at(0, 0, 0, 0) - 1e-4, sum);
  EXPECT_GE(this->blob_top_vec_[0]->data_at(0, 0, 0, 0) + 1e-4, sum); 
}

TYPED_TEST(L1NormLossLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  L1NormLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0, -1, -1);
}
/* //gpu version not implemented yet
TYPED_TEST(L1NormLossLayerTest, TestGradientGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  L1NormLossLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &this->blob_top_vec_);
  GradientChecker<TypeParam> checker(1e-2, 1e-2, 1701);
  checker.CheckGradientSingle(layer, this->blob_bottom_vec_,
      this->blob_top_vec_, 0, -1, -1);
} */

}
