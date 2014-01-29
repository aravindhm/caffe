// Copyright 2013 Yangqing Jia

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#include <iostream>

namespace caffe {

template <typename Dtype>
void CroppingLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  STARTX_ = this->layer_param_.cropx() - 1; //Indexed at 0 not 1
  STARTY_ = this->layer_param_.cropy() - 1; //Indexed at 0 not 1
  HEIGHT_OUT_ = this->layer_param_.crop_height();
  WIDTH_OUT_ = this->layer_param_.crop_width();
  CHECK_EQ(bottom.size(), 1) << "Cropping Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 1) << "Cropping Layer takes a single blob as output.";
  NUM_ = bottom[0]->num();
  CHANNEL_ = bottom[0]->channels();
  HEIGHT_IN_ = bottom[0]->height();
  WIDTH_IN_ = bottom[0]->width();
  (*top)[0]->Reshape(NUM_, CHANNEL_, HEIGHT_OUT_, WIDTH_OUT_);

};

template <typename Dtype>
void CroppingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  const Dtype* bottom_data = bottom[0]->cpu_data();
  // In short, top[n, c, h, w] = bottom[n, c, h+starty, w+startx] if in range
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNEL_; ++c) {
      for (int h = 0; h < HEIGHT_OUT_; ++h) {
        // copy the width part
        memcpy(
            top_data + ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h)
                * WIDTH_OUT_,
            bottom_data + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h + STARTY_) * WIDTH_IN_ + STARTX_,
            sizeof(Dtype) * WIDTH_OUT_);
      }
    }
  }
}

template <typename Dtype>
Dtype CroppingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = (*bottom)[0]->mutable_cpu_diff();
  memset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count());
  // In short,  bottom[n, c, h+starty-1, w+startx-1] = top[n, c, h, w] if in range
  for (int n = 0; n < NUM_; ++n) {
    for (int c = 0; c < CHANNEL_; ++c) {
      for (int h = 0; h < HEIGHT_OUT_; ++h) {
        // copy the width part
        memcpy(
            bottom_diff + ((n * CHANNEL_ + c) * HEIGHT_IN_ + h + STARTY_) * WIDTH_IN_ + STARTX_,
            top_diff + ((n * CHANNEL_ + c) * HEIGHT_OUT_ + h)
                * WIDTH_OUT_,
            sizeof(Dtype) * WIDTH_OUT_);
      }
    }
  }
  return Dtype(0.);
}

template <typename Dtype>
__global__ void CroppingForward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int startx, const int starty, const int height_out, const int width_out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    int w = index % width_out;
    index /= width_out;
    int h = index % height_out;
    index /= height_out;
    int c = index % channel;
    index /= channel;
    out[((index * channel + c) * height_out + h) * width_out + w] =
        in[((index * channel + c) * height_in + h + starty) * width_in + w + startx];
  }
}

template <typename Dtype>
void CroppingLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  const int count = (*top)[0]->count();
  // First, set all data to be zero for the boundary pixels
  CroppingForward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      count, bottom_data, top_data, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
      STARTX_, STARTY_, HEIGHT_OUT_, WIDTH_OUT_);
  CUDA_POST_KERNEL_CHECK;
}

template <typename Dtype>
__global__ void CroppingBackward(const int count, const Dtype* in, Dtype* out,
    const int num, const int channel, const int height_in, const int width_in,
    const int startx, const int starty, const int height_out, const int width_out) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    int w = index % width_out;
    index /= width_out;
    int h = index % height_out;
    index /= height_out;
    int c = index % channel;
    index /= channel;
    out[((index * channel + c) * height_in + h + starty) * width_in + w + startx] =
        in[((index * channel + c) * height_out + h) * width_out + w];
  }
}

template <typename Dtype>
Dtype CroppingLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const bool propagate_down,
    vector<Blob<Dtype>*>* bottom) {
  if (propagate_down) {
    const Dtype* top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = (*bottom)[0]->mutable_gpu_diff();
    const int count = top[0]->count();
    CUDA_CHECK(cudaMemset(bottom_diff, 0, sizeof(Dtype) * (*bottom)[0]->count()));
    CroppingBackward<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
        count, top_diff, bottom_diff, NUM_, CHANNEL_, HEIGHT_IN_, WIDTH_IN_,
        STARTX_, STARTY_, HEIGHT_OUT_, WIDTH_OUT_);
    CUDA_POST_KERNEL_CHECK;
  }
  return Dtype(0);
}

INSTANTIATE_CLASS(CroppingLayer);


}  // namespace caffe
