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
  
template <typename Dtype>
class MeanSubtractLayerLayerTest : public ::testing::Test {
 protected:
  MeanSubtractLayerLayerTest()
      : blob_bottom_(new Blob<Dtype>(2, 3, 4, 5)),
        blob_top_(new Blob<Dtype>()) {
    // fill the values
    FillerParameter filler_param;
    UniformFiller<Dtype> filler(filler_param);
    filler.Fill(this->blob_bottom_);
    blob_bottom_vec_.push_back(blob_bottom_);
    blob_top_vec_.push_back(blob_top_);
  };
  virtual ~MeanSubtractLayerLayerTest() { delete blob_bottom_; delete blob_top_; }
  Blob<Dtype>* const blob_bottom_;
  Blob<Dtype>* const blob_top_;
  vector<Blob<Dtype>*> blob_bottom_vec_;
  vector<Blob<Dtype>*> blob_top_vec_;
};

typedef ::testing::Types<float, double> Dtypes;
TYPED_TEST_CASE(MeanSubtractLayerLayerTest, Dtypes);

TYPED_TEST(MeanSubtractLayerLayerTest, TestSetUp) {
  LayerParameter layer_param;
  layer_param.set_whitening_matrix_file("testwhiteningblob.binaryproto");
  shared_ptr<MeanSubtractLayer<TypeParam> > layer(
  	new MeanSubtractLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  EXPECT_EQ(this->blob_top_->num(), 2);
  EXPECT_EQ(this->blob_top_->channels(), 3);
  EXPECT_EQ(this->blob_top_->height(), 4);
  EXPECT_EQ(this->blob_top_->width(), 5);
}

TYPED_TEST(MeanSubtractLayerLayerTest, TestCPU) {
  LayerParameter layer_param;
  Caffe::set_mode(Caffe::CPU);
  shared_ptr<MeanSubtractLayer<TypeParam> > layer(
  	new MeanSubtractLayer<TypeParam>(layer_param));
  layer->SetUp(this->blob_bottom_vec_, &(this->blob_top_vec_));
  layer->Forward(this->blob_bottom_vec_, &(this->blob_top_vec_));
  const TypeParam* data = this->blob_top_->cpu_data();
  const int count = this->blob_top_->count();
  TypeParam data_in[60][2];
  TypeParam data_out[60][2];

  for(int n = 0; n < this->blob_bottom_->num(); n++) {
    TypeParam mean = 0.0;
    int c,h,w;
    for(c = 0; c < this->blob_bottom_->channels(); c++) {
      for(h = 0; h < this->blob_bottom_->height(); h++) {
        for(w = 0; w < this->blob_bottom_->width(); w++) {
           mean = mean + this->blob_bottom_->data_at(n,c,h,w);
        }
      } 
    }
    mean = mean/(c*h*w);
    int counter = 0;
    for(c = 0; c < this->blob_bottom_->channels(); c++) {
      for(h = 0; h < this->blob_bottom_->height(); h++) {
        for(w = 0; w < this->blob_bottom_->width(); w++) {
          data_out[counter][n] = this->blob_bottom_->data_at(n,c,h,w) - mean;
          counter++;
        }
      } 
    }
  }
  for(int n = 0; n < this->blob_bottom_->num(); n++) {
  int counter = 0;
  for(int c = 0; c < this->blob_bottom_->channels(); c++) {
    for(int h = 0; h < this->blob_bottom_->height(); h++) {
      for(int w = 0; w < this->blob_bottom_->width(); w++) {
        EXPECT_NEAR(data_out[counter][n], this->blob_top_->data_at(n,c,h,w), 1e-4);
      }
    } 
  }
  }
}

} //namespace caffe;
