#include <iostream>
#include <fstream>

#include <glog/logging.h>
#include <leveldb/db.h>
#include <stdint.h>

#include <string>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using caffe::Datum;
using caffe::BlobProto;

//PCAWhitetobinaryproto PCAwhite.txt output.binaryproto

#define DIM (84*84*3)

using namespace std;

int main(int argc, char* argv[]) {
  ::google::InitGoogleLogging(argv[0]);
  if(argc != 3) {
    cerr << "Usage: PCAWhitetobinaryproto PCAwhite.txt output.binaryproto";
    return -1;
  }  
  BlobProto PCAwhite;
  PCAwhite.set_num(1);
  PCAwhite.set_channels(1);
  PCAwhite.set_height(DIM);
  PCAwhite.set_width(DIM);
  fstream fin;
  fin.open(argv[1], ios::in);
  if(!fin.good()) {
    cerr << "Error opening file " << argv[1];
    return -1;
  }
  float val;
  for(int i = 0; i < DIM; i++) {
    for(int j = 0; j < DIM; j++) {
      fin >> val;
      PCAwhite.add_data(val);
    }
  }
  fin.close();
  WriteProtoToBinaryFile(PCAwhite, argv[2]);
  return 0;
}
