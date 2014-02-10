//Copyright 2013 Aravindh Mahendran
//
// This script will copy trained layers from network 1 to network 2. 
// It copies only those layers that have the same name.
// Usage:
//     copy_relevant_layers network2_prototxt pretrained_network1 destination_filename

#include <cuda_runtime.h>

#include <cstring>

#include "caffe/caffe.hpp"

using namespace caffe;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if (argc < 3) {
    LOG(ERROR) << "Usage: copy_relevant_layers network2_prototxt pretrained_network1 destination_filename";
    return 0;
  }

  //read config for network 2
  NetParameter net_param2;
  ReadProtoFromTextFile(argv[1], &net_param2);

  //copy trained layers into network 1
  NetParameter trained_net_param1;
  ReadProtoFromBinaryFile(argv[2], &trained_net_param1);
  
  //Copy the trained layer information into a new network
  Net<float> network2(net_param2);
  network2.CopyTrainedLayersFrom(trained_net_param1);
   
  //Save this for future use
  network2.ToProto(&net_param2, false);
  WriteProtoToBinaryFile(net_param2, argv[3]);
  return 0;
}
