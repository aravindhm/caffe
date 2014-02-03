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
        blob_bottom_label_(new Blob<Dtype>(10, 5, 1, 1)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    GaussianFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_data_);
    blob_bottom_vec_.push_back(blob_bottom_data_);
    filler.Fill(this->blob_bottom_label_);
    blob_bottom_vec_.push_back(blob_bottom_label_);
    blob_top_vec_.push_back(blob_top_);
  }
  virtual ~EuclideanLayerTest() {
    delete blob_bottom_data_;
    delete blob_bottom_label_;
    delete blob_top_;
  }
  Blob<Dtype>* const blob_bottom_data_;
  Blob<Dtype>* const blob_bottom_label_;
  Blob<Dtype>* blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(EuclideanLayerTest, Dtypes);

TYPED_TEST(EuclideanLayerTest, TestSetUp) {
  LayerParameter layer_param;
  shared_ptr<EuclideanLayer<TypeParam> > layer(
  	new EuclideanLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 10);
  EXPECT_EQ(this->blob_top_->height(), 1);
  EXPECT_EQ(this->blob_top_->width(), 1);
  EXPECT_EQ(this->blob_top_->channels(), 1);
}


TYPED_TEST(EuclideanLayerTest, TestCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  shared_ptr<EuclideanLayer<TypeParam> > layer(
  	new EuclideanLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
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
  shared_ptr<EuclideanLayer<TypeParam> > layer(
  	new EuclideanLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
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

TYPED_TEST(EuclideanLayerTest, TestGradientCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  EuclideanLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}

TYPED_TEST(EuclideanLayerTest, TestGradientGPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::GPU);
  EuclideanLayer<TypeParam> layer(layer_param);
  GradientChecker<TypeParam> checker(1e-2, 1e-2);
  checker.CheckGradientExhaustive(layer, this->blob_bottom_vec_,
      this->blob_top_vec_);
}
}
