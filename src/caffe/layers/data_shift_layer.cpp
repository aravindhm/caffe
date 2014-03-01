// Copyright 2014 Jeff Donahue

#include <vector>

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DataShiftLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Data shift Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 2) << "Data shift Layer takes at least one blob as output.";
  CHECK_GE(bottom[0]->num(), 2) << "Input to data shift should have a batch size more than 1";
  //we will delete one of the data points in the batch so set count_ accordingly
  offset_ = bottom[0]->channels()*bottom[0]->height()*bottom[0]->width();
  count_ = bottom[0]->count() - offset_;
  for (int i = 0; i < top->size(); ++i) {
    CHECK_NE((*top)[i], bottom[0]) << "Blobs cannot be in place for the data shift layer";
    (*top)[i]->Reshape(bottom[0]->num()-1, bottom[0]->channels(),
                       bottom[0]->height(), bottom[0]->width());
    CHECK_EQ(count_, (*top)[i]->count());
  }
};

template <typename Dtype>
void DataShiftLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data1 = (*top)[0]->mutable_cpu_data();
  Dtype* top_data2 = (*top)[1]->mutable_cpu_data();
  caffe_copy(count_, bottom_data, top_data1);
  caffe_copy(count_, bottom_data+offset_, top_data2);
}

template <typename Dtype>
void DataShiftLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data1 = (*top)[0]->mutable_gpu_data();
  Dtype* top_data2 = (*top)[1]->mutable_gpu_data();
  caffe_gpu_copy(count_, bottom_data, top_data1);
  caffe_gpu_copy(count_, bottom_data+offset_, top_data2);
}

template <typename Dtype>
Dtype DataShiftLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}


template <typename Dtype>
Dtype DataShiftLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(DataShiftLayer);

}  // namespace caffe
