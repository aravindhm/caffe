// Copyright 2013 Aravindh Mahendran

#ifndef CAFFE_VISION_REGULARIZERS_HPP_
#define CAFFE_VISION_REGULARIZERS_HPP_

#include <leveldb/db.h>
#include <pthread.h>

#include <vector>

#include "caffe/regularizer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

/*
template <typename Dtype>
class FeaturesL1Regularizer : public Regularizer<Dtype> {
  public: 
    explicit FeaturesL1Regularizer(const RegularizerParameter& param) 
      : Regularizer<Dtype>(param) {}
  protected: 
    virtual Dtype Regularize_cpu(vector<shared_ptr<Blob<Dtype> > >& blobs, 
       const bool propagate_down, 
       vector<Blob<Dtype>*>* bottom,
       const vector<Blob<Dtype>*>& top);
}; // class FeaturesL1Regularizer
*/
} //namespace caffe

#endif
