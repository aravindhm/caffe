// Copyright 2013 Aravindh Mahendran
#include <glog/logging.h>
#include <leveldb/db.h>
#include <stdint.h>

#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <iostream>

using caffe::Datum;
using caffe::BlobProto;

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if( argc != 2 ) {
    LOG(ERROR) << "Usage: view_mean meanfile.binaryproto";
    return (0);
  }
  BlobProto sum_blob;
  ReadProtoFromBinaryFile(argv[1], &sum_blob);
  for(int i =0 ; i < sum_blob.data_size(); i++) {
    std::cout << sum_blob.data(i) << " ";
  } 
  std::cout << std::endl;
  return 0;
}
