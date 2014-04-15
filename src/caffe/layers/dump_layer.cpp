// Copyright 2014 Aravindh Mahendran
// adapted from Padding layer code

#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"

#include <iostream>
#include <fstream>
#include <sstream>

namespace caffe {

template <typename Dtype>
void DumpLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  CHECK_EQ(bottom.size(), 1) << "Dump Layer takes a single blob as input.";
  CHECK_EQ(top->size(), 0) << "Dump Layer takes no output.";
  iter_ = 0;
};

template <typename Dtype>
void DumpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  std::stringstream ss;
  ss << iter_ << ".txt";
  std::fstream fout;
  fout.open(ss.str().c_str(), std::ios::out);
  if(!fout.good() ) {
    LOG(FATAL) << "Failed to open dump file " << ss.str();
  }
  for(int i = 0; i < bottom[0]->count(); i++) {
    fout << bottom_data[i] << " ";
  }
  fout.close();
  iter_++;
}

INSTANTIATE_CLASS(DumpLayer);


}  // namespace caffe
