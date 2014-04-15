// Copyright 2014 Aravindh Mahendran


//#include <mkl.h>
#include <cublas_v2.h>

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/filler.hpp"
#include "caffe/layer.hpp"
#include "caffe/vision_layers.hpp"
#include "caffe/util/math_functions.hpp"

#include "caffe/util/io.hpp"

namespace caffe {

template <typename Dtype>
void WhiteningLayer<Dtype>::SetUp(const vector<Blob<Dtype>*>& bottom,
      vector<Blob<Dtype>*>* top) {
 CHECK_EQ(bottom.size(), 1) << "Whitening Layer takes a single blob as input.";
 CHECK_EQ(top->size(), 1) << "Whitening Layer takes a single blob as output.";
  if(this->layer_param_.has_whitening_matrix_file()) {
    BlobProto blob_proto;
    LOG(INFO) << "Loading whitening matrix from " << this-> layer_param_.whitening_matrix_file();
    ReadProtoFromBinaryFile(this->layer_param_.whitening_matrix_file().c_str(), &blob_proto);
    whitening_matrix_.FromProto(blob_proto);
    CHECK_EQ(whitening_matrix_.num(), 1);
    CHECK_EQ(whitening_matrix_.channels(), 1);
    CHECK_EQ(whitening_matrix_.width(), bottom[0]->count()/bottom[0]->num());
    CHECK_EQ(whitening_matrix_.height(), bottom[0]->count()/bottom[0]->num());

    M_ = bottom[0]->num();
    K_ = bottom[0]->count() / bottom[0]->num();
    N_ = K_;

    (*top)[0]->Reshape(bottom[0]->num(), bottom[0]->channels(), 
       bottom[0]->height(), bottom[0]->width());
  }
  else {
    LOG(FATAL) << "Cannot without whitening matrix. Please specify whitening matrix in prototxt";
  }
}

template <typename Dtype>
void WhiteningLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* bottom_data_mutable = bottom[0]->mutable_cpu_data();
  Dtype* top_data = (*top)[0]->mutable_cpu_data();
  int dim = bottom[0]->count()/bottom[0]->num();
  // First subtract the mean
  for(int i = 0; i < bottom[0]->num(); i++) {
    Dtype mean = caffe_cpu_mean(dim, bottom_data + i*dim);
    caffe_sub(dim, bottom_data+i*dim, mean, bottom_data_mutable+i*dim);
  }
  const Dtype* whitening_matrix_data = this->whitening_matrix_.cpu_data();
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data, whitening_matrix_data, (Dtype)0., top_data);
}

template <typename Dtype>
void WhiteningLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    vector<Blob<Dtype>*>* top) {
  const Dtype* bottom_data_cpu = bottom[0]->cpu_data();
  Dtype* bottom_data_mutable_cpu = bottom[0]->mutable_cpu_data();
  Dtype* top_data = (*top)[0]->mutable_gpu_data();
  int dim = bottom[0]->count()/bottom[0]->num();
  // First subtract the mean
  for(int i = 0; i < bottom[0]->num(); i++) {
    Dtype mean = caffe_cpu_mean(dim, bottom_data_cpu + i*dim);
    caffe_sub(dim, bottom_data_cpu+i*dim, mean, bottom_data_mutable_cpu+i*dim);
  }
  const Dtype* bottom_data_gpu = bottom[0]->gpu_data();
  const Dtype* whitening_matrix_data = this->whitening_matrix_.gpu_data();
  caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
      bottom_data_gpu, whitening_matrix_data, (Dtype)0., top_data);
}

// The backward operations are dummy - they do not carry any computation.
template <typename Dtype>
Dtype WhiteningLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

template <typename Dtype>
Dtype WhiteningLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
      const bool propagate_down, vector<Blob<Dtype>*>* bottom) {
  return Dtype(0.);
}

INSTANTIATE_CLASS(WhiteningLayer);

} // namespace caffe
