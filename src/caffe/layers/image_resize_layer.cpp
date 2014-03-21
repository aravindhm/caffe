// Copyright 2014 Aravindh Mahendran
// adapted from Padding layer code

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#include <opencv2/opencv.hpp>

#include <iostream>

namespace caffe {

template <typename Dtype>
void ImageResizeLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  HEIGHT_OUT_ = this->layer_param_.target_height();
  WIDTH_OUT_ = this->layer_param_.target_width();
  CHECK_EQ(bottom.size(), 1) << "ImageResize Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "ImageResize Layer takes a single blob as output.";
  NUM_ = bottom[0]->num();
  CHANNEL_ = bottom[0]->channels();
  CHECK_EQ(CHANNEL_, 3) << "ImageResize can only operate on 3 channel images";
  HEIGHT_IN_ = bottom[0]->height();
  WIDTH_IN_ = bottom[0]->width();
  LOG(INFO) << "Image resize top reshape";
  (*top)[0]->Reshape(NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_);
  LOG(INFO) << "Image resize setup complete";
};

template <typename Dtype>
void ImageResizeLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  LOG(INFO) << "Image resize Forward started";
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  cv::Mat img(HEIGHT_OUT_, WIDTH_OUT_, CV_32FC(3));
  cv::Mat img_resized;
  // In short, bottom -> img -> img_resized -> top
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNEL_; ++c) {
      LOG(INFO) << "Channel " << c;
      for (int h = 0; h < HEIGHT_IN_; ++h) {
        for (int w = 0; w < WIDTH_IN_; ++w) {
          LOG(INFO) << h << " " << w;
          img.at<cv::Vec3f>(h,w)[c] = *(bottom_data + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h) * WIDTH_IN_ + w);
        }
      }
    }
    LOG(INFO) << "Done with the data copy for " << n;
    cv::resize(img, img_resized, cv::Size(HEIGHT_OUT_, WIDTH_OUT_), 0, 0);
    LOG(INFO) << "Done with the resize for " << n;
    for (int c = 0; c < CHANNEL_; ++c) {
      LOG(INFO) << "Channel " << c;
      for (int h = 0; h < HEIGHT_OUT_; ++h) {
        for (int w = 0; w < WIDTH_OUT_; ++w) {
          *(top_data + ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h) * WIDTH_OUT_ + w)
             = img_resized.at<cv::Vec3f>(h,w)[c];
        }
      }
    }
  }
  LOG(INFO) << "Images resized";
}

template <typename Dtype>
Dtype ImageResizeLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(ImageResizeLayer);


}  // namespace caffe
