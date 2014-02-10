// Copyright 2013 Aravindh Mahendran

#ifndef CAFFE_REGULARIZER_FACTORY_HPP_
#define CAFFE_REGULARIZER_FACTORY_HPP_

#include <string>

#include "caffe/regularizer.hpp"
#include "caffe/vision_regularizers.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

// A function to get a specific layer from the specification given in
// LayerParameter. Ideally this would be replaced by a factory pattern,
// but we will leave it this way for now.
template <typename Dtype>
Regularizer<Dtype>* GetRegularizer(const RegularizerParameter& param) {
  const std::string& type = param.type();
  if (type == "features_l1") {
    return new FeatureL1Regularizer<Dtype>(param);
/*  } else if (type == "weights_l1") {
    return new WeightL1Regularizer<Dtype>(param);*/
  } else {
    LOG(FATAL) << "Unknown regularizer name: " << type;
  }
  // just to suppress old compiler warnings.
  return (Regularizer<Dtype>*)(NULL);
}

template Regularizer<float>* GetRegularizer(const RegularizerParameter& param);
template Regularizer<double>* GetRegularizer(const RegularizerParameter& param);

}  // namespace caffe

#endif
