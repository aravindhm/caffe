// Copyright 2013 Yangqing Jia

#include <cmath>
#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/test/test_gradient_check_util.hpp"

#include "caffe/test/test_caffe_main.hpp"

template <typename Dtype>
Dtype zerothresh(Dtype num) {
   if(num > 0) return num;
   else return Dtype(0.);
}

namespace caffe {

extern cudaDeviceProp CAFFE_TEST_CUDA_PROP;

template <typename Dtype>
class ReLULayerTest : public ::testing::Test {
 protected:
  ReLULayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 10, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~ReLULayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(ReLULayerTest, Dtypes);

TYPED_TEST(ReLULayerTest, TestForwardCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ReLULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Test exact values
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
      for (int k = 0; k < this->blob_bottom_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom_->width(); ++l) {
          EXPECT_GE(this->blob_top_->data_at(i,j,k,l) + 1e-4,
             zerothresh<TypeParam>(this->blob_bottom_->data_at(i,j,k,l)));
          EXPECT_LE(this->blob_top_->data_at(i,j,k,l) - 1e-4,
             zerothresh<TypeParam>(this->blob_bottom_->data_at(i,j,k,l)));
        }
      }
    }
  }
}

TYPED_TEST(ReLULayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  ReLULayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(ReLULayerTest, TestForwardGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  ReLULayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  // Test exact values
  for (int i = 0; i < this->blob_bottom_->num(); ++i) {
    for (int j = 0; j < this->blob_bottom_->channels(); ++j) {
      for (int k = 0; k < this->blob_bottom_->height(); ++k) {
        for (int l = 0; l < this->blob_bottom_->width(); ++l) {
          EXPECT_GE(this->blob_top_->data_at(i,j,k,l) + 1e-4,
             zerothresh(this->blob_bottom_->data_at(i,j,k,l)));
          EXPECT_LE(this->blob_top_->data_at(i,j,k,l) - 1e-4,
             zerothresh(this->blob_bottom_->data_at(i,j,k,l)));
        }
      }
    }
  }
}

TYPED_TEST(ReLULayerTest, TestGradientGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  ReLULayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

}
