// Copyright 2014 Aravindh Mahendran
//
// This is a simple script that allows one to quickly run the covariance computing  network and dump results into a file
//    demo_compute_covariance net_proto iterations outfile

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>

#include <fstream>

#include "caffe/caffe.hpp"

using namespace caffe;
using namespace std;

int main(int argc, char** argv) {
  if (argc < 4) {
    LOG(ERROR) << "demo_compute_covariance net_proto iterations outfile";
    return 0;
  }

  cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);

  Caffe::set_mode(Caffe::GPU);

  NetParameter test_net_param;
  ReadProtoFromTextFile(argv[1], &test_net_param);
  Net<float> caffe_test_net(test_net_param);

  int total_iter = atoi(argv[2]);
  LOG(ERROR) << "Running " << total_iter << "Iterations.";

  fstream outfile;
  outfile.open(argv[3], ios::out);
  if(!outfile.good()) {
    LOG(FATAL) << "Error opening outfile " << argv[3];
  }

  vector<Blob<float>*> dummy_blob_input_vec;
  for (int i = 0; i < total_iter - 1; ++i) {
    LOG(INFO) << "Iteration: " << i;
    const vector<Blob<float>*>& result =
        caffe_test_net.Forward(dummy_blob_input_vec);
  }
  //final iteration
  const vector<Blob<float>*>& result =
      caffe_test_net.Forward(dummy_blob_input_vec);
  
  for(int h = 0; h < result[1]->height(); h++) {
    for(int w = 0; w < result[1]->width(); w++) {
      outfile << result[1]->data_at(0,0,h,w) << " ";
    }
    outfile << endl;
  }
  outfile.close();
  return 0;
}
