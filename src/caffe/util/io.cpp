// Copyright 2013 Yangqing Jia

#include <stdint.h>
#include <fcntl.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/coded_stream.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>

#include "caffe/common.hpp"
#include "caffe/util/io.hpp"
#include "caffe/proto/caffe.pb.h"

using std::fstream;
using std::ios;
using std::max;
using std::string;
using std::vector;
using std::cout;
using std::endl;

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedOutputStream;

namespace caffe {

void ReadProtoFromTextFile(const char* filename,
    ::google::protobuf::Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  FileInputStream* input = new FileInputStream(fd);
  CHECK(google::protobuf::TextFormat::Parse(input, proto));
  delete input;
  close(fd);
}

void WriteProtoToTextFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY);
  FileOutputStream* output = new FileOutputStream(fd);
  CHECK(google::protobuf::TextFormat::Print(proto, output));
  delete output;
  close(fd);
}

void ReadProtoFromBinaryFile(const char* filename, Message* proto) {
  int fd = open(filename, O_RDONLY);
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(536870912, 268435456);

  CHECK(proto->ParseFromCodedStream(coded_input));

  delete coded_input;
  delete raw_input;
  close(fd);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  fstream output(filename, ios::out | ios::trunc | ios::binary);
  CHECK(proto.SerializeToOstream(&output));
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum) {
  cv::Mat cv_img;
  if (height > 0 && width > 0) {
    cv::Mat cv_img_origin = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
    cv::resize(cv_img_origin, cv_img, cv::Size(height, width));
  } else {
    cv_img = cv::imread(filename, CV_LOAD_IMAGE_COLOR);
  }
  if (!cv_img.data) {
    LOG(ERROR) << "Could not open or find file " << filename;
    return false;
  }
  if (height > 0 && width > 0) {

  }
  datum->set_channels(3);
  datum->set_height(cv_img.rows);
  datum->set_width(cv_img.cols);
  datum->set_label(label);
  datum->clear_data();
  datum->clear_float_data();
  string* datum_string = datum->mutable_data();
  for (int c = 0; c < 3; ++c) {
    for (int h = 0; h < cv_img.rows; ++h) {
      for (int w = 0; w < cv_img.cols; ++w) {
        datum_string->push_back(static_cast<char>(cv_img.at<cv::Vec3b>(h, w)[c]));
      }
    }
  }
  return true;
}

/* Read images given a vector of filenames into a vector of cv::Mat */
vector<boost::shared_ptr<cv::Mat> > ReadImagesToCvMat(const vector<string>& filenames) {
  vector<boost::shared_ptr<cv::Mat> > images;
  for(int i = 0; i < filenames.size(); i++) {
    boost::shared_ptr<cv::Mat> img_ptr(new cv::Mat());
    *(img_ptr.get()) = cv::imread(filenames[i]);
    if(img_ptr->empty()) {
      LOG(FATAL) << "Failed to load image " << filenames[i];
    }
    images.push_back(img_ptr);
  }
  return images;
}

/* Convert a blob into color mapped images */
vector<boost::shared_ptr<cv::Mat> > Blob2ColorMap(const boost::shared_ptr<Blob<float> > blob) {
  vector<boost::shared_ptr<cv::Mat> > color_maps;
  double min;
  double max;

  for(int n = 0; n < blob->num(); n++) {
    for(int c = 0; c < blob->channels(); c++) {
       //copy the data into an opencv matrix
       cv::Mat img;
       cv::Mat adjMap;
       boost::shared_ptr<cv::Mat> color_img(new cv::Mat());
       img.create(blob->height(), blob->width(), CV_32FC(1));
       for(int h = 0; h < blob->height(); h++) {
         for(int w = 0; w < blob->width(); w++) {
           img.at<float>(h,w) = blob->data_at(n,c,h,w);
         }
       }
       cv::minMaxIdx(img, &min, &max);
       cv::convertScaleAbs(img, adjMap, 255 / max);
       //apply the color map
       cv::applyColorMap(adjMap, *(color_img.get()), cv::COLORMAP_JET);

       //push it into the result
       color_maps.push_back(color_img);
    }
  }
  return color_maps;
}

/* Convert opencv matrix into a blob.
   Works only for 3D matrices. In other words blob.num() is set to 1
   */
boost::shared_ptr<Blob<float> > CvMatToBlob(cv::Mat mat) {
  boost::shared_ptr<Blob<float> > blob(new Blob<float>(1, mat.channels(), mat.rows, mat.cols));
  float* data = blob->mutable_cpu_data();
  for(int c = 0; c < mat.channels(); c++) {
    for(int h = 0; h < mat.rows; h++) {
      for(int w = 0; w < mat.cols; w++) {
        *(data + blob->offset(0, c, h, w)) = mat.at<cv::Vec3b>(h,w)[c];
      }
    }
  }
  return blob;
}

}  // namespace caffe
