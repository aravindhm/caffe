// Copyright Yangqing Jia 2013

#ifndef CAFFE_UTIL_IO_H_
#define CAFFE_UTIL_IO_H_

#include <google/protobuf/message.h>

#include <string>

#include "caffe/blob.hpp"
#include "caffe/proto/caffe.pb.h"

#include <opencv2/opencv.hpp>
#include <opencv2/contrib/contrib.hpp>

#include <vector>

using std::string;
using ::google::protobuf::Message;

namespace caffe {

void ReadProtoFromTextFile(const char* filename,
    Message* proto);
inline void ReadProtoFromTextFile(const string& filename,
    Message* proto) {
  ReadProtoFromTextFile(filename.c_str(), proto);
}

void WriteProtoToTextFile(const Message& proto, const char* filename);
inline void WriteProtoToTextFile(const Message& proto, const string& filename) {
  WriteProtoToTextFile(proto, filename.c_str());
}

void ReadProtoFromBinaryFile(const char* filename,
    Message* proto);
inline void ReadProtoFromBinaryFile(const string& filename,
    Message* proto) {
  ReadProtoFromBinaryFile(filename.c_str(), proto);
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename);
inline void WriteProtoToBinaryFile(
    const Message& proto, const string& filename) {
  WriteProtoToBinaryFile(proto, filename.c_str());
}

bool ReadImageToDatum(const string& filename, const int label,
    const int height, const int width, Datum* datum);

inline bool ReadImageToDatum(const string& filename, const int label,
    Datum* datum) {
  return ReadImageToDatum(filename, label, 0, 0, datum);
}

std::vector<cv::Mat> ReadImagesToCvMat(const std::vector<string>& filenames);

std::vector<cv::Mat> Blob2ColorMap(const shared_ptr<Blob<float> > blob);

shared_ptr<Blob<float> > CvMatToBlob(cv::Mat mat);

}  // namespace caffe

#endif   // CAFFE_UTIL_IO_H_
