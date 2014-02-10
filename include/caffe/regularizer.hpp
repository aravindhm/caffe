//Copyright 2013 Aravindh Mahendran

#ifndef CAFFE_REGULARIZER_H_
#define CAFFE_REGULARIZER_H_

#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/proto/caffe.pb.h"

using std::vector;

namespace caffe {

template <typename Dtype>
class Regularizer {
  public:
    //constructor does nothing but initialize the parameters
    explicit Regularizer(const RegularizerParameter& param)
      : regularizer_param_(param) {}
    virtual ~Regularizer() {}
    
    //this is the main function called by the layer class. It should call everything else
    inline Dtype Regularize(vector<shared_ptr<Blob<Dtype> > >& blobs, 
        const bool propagate_down, 
        vector<Blob<Dtype>*>* bottom,
        const vector<Blob<Dtype>*>& top) {
      switch (Caffe::mode()) {
      case Caffe::CPU:
        return Regularize_cpu(blobs, propagate_down, bottom, top);
      case Caffe::GPU:
        return Regularize_gpu(blobs, propagate_down, bottom, top);
      default:
        LOG(FATAL) << "Unknown caffe mode.";
      }
      return Dtype(0.);
    }
    
    //accessor for the regularization parameters 
    const RegularizerParameter& regularizer_param() { return regularizer_param_; }

  protected:
    // The protobuf that stores the regularizer parameters
    RegularizerParameter regularizer_param_;
    
    // Regularize function for cpu mode and gpu mode
    virtual Dtype Regularize_cpu(vector<shared_ptr<Blob<Dtype> > >& blobs, 
        const bool propagate_down, 
        vector<Blob<Dtype>*>* bottom,
        const vector<Blob<Dtype>*>& top) = 0;
    //The gpu version is not made pure virtual. It simply calls the cpu version 
    // unless the base class overrides it.
    virtual Dtype Regularize_gpu(vector<shared_ptr<Blob<Dtype> > >& blobs, 
        const bool propagate_down, 
        vector<Blob<Dtype>*>* bottom,
        const vector<Blob<Dtype>*>& top) {
        return Regularize_cpu(blobs, propagate_down, bottom, top);
    }
    
    DISABLE_COPY_AND_ASSIGN(Regularizer);
}; // class Regularizer

// The regularizer factory function
template <typename Dtype>
Regularizer<Dtype>* GetRegularizer(const RegularizerParameter& param);

} //namespace caffe

#endif
