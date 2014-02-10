// Copyright 2013 Aravindh Mahendran

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <vector>
#include <thrust/device_vector.h>

#include <eigen3/Eigen/Core>

#include "caffe/regularizer.hpp"
#include "caffe/vision_regularizers.hpp"
#include "caffe/util/math_functions.hpp"

#include <iostream>

namespace caffe {

template <typename Dtype>
Dtype FeatureL1Regularizer<Dtype>::Regularize_cpu(vector<shared_ptr<Blob<Dtype> > >& blobs, 
          const bool propagate_down, 
          vector<Blob<Dtype>*>* bottom,
          const vector<Blob<Dtype>*>& top) {
    return Dtype(0.);
}

} //namespace caffe
