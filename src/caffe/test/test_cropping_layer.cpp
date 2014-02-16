// Copyright 2014 Aravindh Mahendran

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
class CroppingLayerTest : public ::testing::Test {
 protected:
  CroppingLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~CroppingLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(CroppingLayerTest, Dtypes);

TYPED_TEST(CroppingLayerTest, TestCPU) {
  LayerParameter layer_param;
  layer_param.set_cropx(1);
  layer_param.set_cropy(1);
  layer_param.set_crop_width(5);
  layer_param.set_crop_height(4);
  Caffe::set_mode(Caffe::CPU);
  CroppingLayer<TypeParam> layer(layer_param);
  layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
  for (int n = 0; n < 2; ++n) {
    for (int c = 0; c < 3; ++c) {
      for (int h = 0; h < 4; ++h) {
        for (int w = 0; w < 5; ++w) {
          EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
              this->blob_top_->data_at(n, c, h, w));
        }
      }
    }
  }
}

TYPED_TEST(CroppingLayerTest, TestCPUGrad) {
  LayerParameter layer_param;
  layer_param.set_cropx(1);
  layer_param.set_cropy(1);
  layer_param.set_crop_width(5);
  layer_param.set_crop_height(4);
  Caffe::set_mode(Caffe::CPU);
  CroppingLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-3);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
}

TYPED_TEST(CroppingLayerTest, TestGPU) {
  if (CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    layer_param.set_cropx(1);
    layer_param.set_cropy(1);
    layer_param.set_crop_width(5);
    layer_param.set_crop_height(4);
    Caffe::set_mode(Caffe::GPU);
    CroppingLayer<TypeParam> layer(layer_param);
    layer.SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
    layer.Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
    EXPECT_EQ(this->blob_top_->num(), 2);
    EXPECT_EQ(this->blob_top_->channels(), 3);
    EXPECT_EQ(this->blob_top_->height(), 4);
    EXPECT_EQ(this->blob_top_->width(), 5);
    for (int n = 0; n < 2; ++n) {
      for (int c = 0; c < 3; ++c) {
        for (int h = 0; h < 4; ++h) {
          for (int w = 0; w < 5; ++w) {
            EXPECT_EQ(this->blob_bottom_->data_at(n, c, h, w),
                this->blob_top_->data_at(n, c, h, w));
          }
        }
      }
    }
  } else {
    LOG(ERROR) << "Skipping test (gpu version too low).";
  }
}

TYPED_TEST(CroppingLayerTest, TestGPUGrad) {
  if (CAFFE_TEST_CUDA_PROP.major >= 2) {
    LayerParameter layer_param;
    layer_param.set_cropx(1);
    layer_param.set_cropy(1);
    layer_param.set_crop_width(5);
    layer_param.set_crop_height(4);
    Caffe::set_mode(Caffe::GPU);
    CroppingLayer<TypeParam> layer(layer_param);
    GradientChecker<TypeParam> checker(1e-2, 1e-3);
    checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_, this->blob_top_vec_);
  } else {
    LOG(ERROR) << "Skipping test (gpu version too low).";
  }
}

}
