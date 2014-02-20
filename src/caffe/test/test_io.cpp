// Copyright 2014 Aravindh Mahendran

#include <cstring>
#include <cuda_runtime.h>

#include "gtest/gtest.h"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"
#include "caffe/filler.hpp"
#include "caffe/util/io.hpp"

#include "caffe/test/test_caffe_main.hpp"

#include "caffe/proto/caffe.pb.h"

#include <vector>

#include <opencv2/opencv.hpp>

#include <iostream>

using namespace std;

namespace caffe {

template <typename Dtype>
class IOSimpleTest : public ::testing::Test {
 protected:
  IOSimpleTest()
      : blob_(new Blob<Dtype>(1, 3, 480, 640)) {}
  virtual ~IOSimpleTest() { delete blob_; }
  Blob<Dtype>* const blob_;
  cv::Mat img;
};

typedef ::testing::Types<float> Dtypes;
TYPED_TEST_CASE(IOSimpleTest, Dtypes);

TYPED_TEST(IOSimpleTest, TestInitialization) {
  EXPECT_TRUE(this->blob_);
  EXPECT_EQ(this->blob_->num(), 1);
  EXPECT_EQ(this->blob_->channels(), 3);
  EXPECT_EQ(this->blob_->height(), 480);
  EXPECT_EQ(this->blob_->width(), 640);
  EXPECT_EQ(this->blob_->count(), 3*480*640);
}

TYPED_TEST(IOSimpleTest, TestBlob2ColorMap) {
/*  FillerParameter param;
  param.set_type("constant");
  param.set_min(0);
  param.set_max(1);
  param.set_value(0.5);
  ConstantFiller<TypeParam> filler(param);
  filler.Fill(this->blob_);*/
  cv::Mat img = cv::imread("/data/kitti/sequences/00/image_2/000000.png");
  boost::shared_ptr<Blob<TypeParam> > blobimg = CvMatToBlob(img);
  std::vector<cv::Mat> colormaps = Blob2ColorMap(blobimg);
  cv::namedWindow("test_io"); 
  for(int i = 0; i < colormaps.size(); i++) {
    cv::imshow("test_io", colormaps[i]);
    cv::waitKey(0);
  }
  cv::destroyWindow("test_io");
}

}
