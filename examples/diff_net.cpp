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
  if (argc < 4) {
    LOG(ERROR) << "view_net net_proto pretrained_net_proto1 pretrained_net_proto2 [CPU/GPU]";
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
  Net<float> caffe_test_net1(test_net_param);
  NetParameter trained_net_param1;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param1);
  caffe_test_net1.CopyTrainedLayersFrom(trained_net_param1);

  Net<float> caffe_test_net2(test_net_param);
  NetParameter trained_net_param2;
  ReadProtoFromBinaryFile(argv[3], &trained_net_param2);
  caffe_test_net2.CopyTrainedLayersFrom(trained_net_param2);

/*  vector<Blob<float>*> dummy_blob_input_vec;
   const vector<Blob<float>*>& result =
        caffe_test_net.Forward(dummy_blob_input_vec);
*/
  vector<shared_ptr<Layer<float> > > net_layers1 = caffe_test_net1.layers();
  vector<shared_ptr<Layer<float> > > net_layers2 = caffe_test_net2.layers();

  for(int i = 0; i < net_layers1.size(); i++) {
    std::cout << net_layers1[i] -> layer_param().name() << " " << i << std::endl;
  }  

  for(int layer_id = 0; layer_id < net_layers1.size(); layer_id++) {
    std::cout << "Layer_id: " << layer_id;
    std::cerr << net_layers1[layer_id] -> blobs().size() << std::endl;
    if(net_layers1[layer_id] -> blobs().size() > 0) {
    for(int i = 0; i < net_layers1[layer_id] -> blobs()[0] -> num(); i++) {
      for(int j = 0; j < net_layers1[layer_id] -> blobs()[0] -> channels(); j++) {
        for(int k = 0; k < net_layers1[layer_id] -> blobs()[0] -> height(); k++) {
          for(int l = 0; l < net_layers1[layer_id] -> blobs()[0] -> width(); l++) {
            std::cout << net_layers2[layer_id] -> blobs()[0] -> data_at(i,j,k,l) - net_layers1[layer_id] -> blobs()[0] -> data_at(i,j,k,l) <<" ";
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
