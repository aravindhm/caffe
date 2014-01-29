// Copyright 2013 Yangqing Jia
//
// This is a simple script that allows one to quickly test a network whose
// structure is specified by text format protocol buffers, and whose parameter
// are loaded from a pre-trained network.
// Usage:
//    view_net net_proto pretrained_net_proto [CPU/GPU]

#include <cuda_runtime.h>

#include <cstring>
#include <cstdlib>

#include "caffe/caffe.hpp"

#include <iostream>

using namespace caffe;

int main(int argc, char** argv) {
  if (argc < 3) {
    LOG(ERROR) << "view_net net_proto pretrained_net_proto [CPU/GPU]";
    return 0;
  }

  cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);

  if (argc == 5 && strcmp(argv[4], "GPU") == 0) {
    LOG(ERROR) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
    LOG(ERROR) << "Using CPU";
    Caffe::set_mode(Caffe::CPU);
  }

  NetParameter test_net_param;
  ReadProtoFromTextFile(argv[1], &test_net_param);
  Net<float> caffe_test_net(test_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);
  caffe_test_net.CopyTrainedLayersFrom(trained_net_param);

/*  vector<Blob<float>*> dummy_blob_input_vec;
   const vector<Blob<float>*>& result =
        caffe_test_net.Forward(dummy_blob_input_vec);
*/
  vector<shared_ptr<Blob<float> > > net_blobs = caffe_test_net.blobs();
  vector<shared_ptr<Layer<float> > > net_layers = caffe_test_net.layers();

  for(int i = 0; i < net_layers.size(); i++) {
    std::cout << net_layers[i] -> layer_param().name() << " " << i << std::endl;
  }  

  for(int layer_id = 0; layer_id < net_layers.size(); layer_id++) {
    std::cout << "Layer_id: " << layer_id;
    std::cerr << net_layers[layer_id] -> blobs().size() << std::endl;
    if(net_layers[layer_id] -> blobs().size() > 0) {
    for(int i = 0; i < net_layers[layer_id] -> blobs()[0] -> num(); i++) {
      for(int j = 0; j < net_layers[layer_id] -> blobs()[0] -> channels(); j++) {
        for(int k = 0; k < net_layers[layer_id] -> blobs()[0] -> height(); k++) {
          for(int l = 0; l < net_layers[layer_id] -> blobs()[0] -> width(); l++) {
            std::cout << net_layers[layer_id] -> blobs()[0] -> data_at(i,j,k,l) <<" ";
          }
          std::cout << std::endl;
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    } 
    }
    std::cout << std::endl;
  }
  return 0;
}
