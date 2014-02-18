// Copyright 2014 Aravindh Mahendran
//
// This program takes a blob proto as input and
// visualizes it as an image and writes it into a file.
// Each channel is visualized separately to give an output similar to matlab's imagesc
//
// blobproto2image blobproto output_image_filename_prefix


#include <google/protobuf/text_format.h>

#include <cstring>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <opencv2/opencv.hpp>

#include <sstream>

using namespace caffe;
using namespace cv;
using namespace std;

int main(int argc, char** argv) {
  if(argc < 2) {
    cerr << "Usage: blobproto2image blobprotofile output_image_filename_prefix";
    return -1;
  }
  BlobProto input_proto;
  ReadProtoFromBinaryFile(argv[0], &input_proto);
  Blob<float> blob;
  blob.FromProto(input_proto);

  Mat img;
  Mat color_img;

  for(int n = 0; n < blob.num(); n++) {
    for(int c = 0; c < blob.channels(); c++) {
       img.create(blob.height(), blob.width(), CV_8UC(1));
       applyColorMap(img, color_img, COLORMAP_JET);
       stringstream ss;
       ss << argv[1] << "_n" << n << "_c" << c << ".png";
       cout << ss.str();
       //imwrite(argv[1], color_img);
    }
  }
  return 0;
}
