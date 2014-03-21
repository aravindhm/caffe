// Copyright 2014 Aravindh Mahendran
// This program converts bounding boxes given by selective search into a leveldb. 
// Note that this does not compute features, just the images are dumped into disk in a compact way.
// Usage:
//     dump_images_in_leveldb csv_files_list DB_NAME
//
// where csv_files_list is the list of csv files containing the bounding boxes
//   DB_NAME is the database to dump the images into


#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>

#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

#include <opencv2/opencv.hpp>

using namespace caffe;
using std::pair;
using std::vector;
using std::string;

void convert_image_to_datum(cv::Mat& cv_img, int label, Datum* datum);

int main(int argc, char** argv) {
  ::google::InitGoogleLogging(argv[0]);
  if(argc < 3) {
    printf(" dump_images_in_leveldb csv_files_list DB_NAME\n");
    return 0;
  }

  std::ifstream fcsv(argv[1]);
  vector<string> lines;
  string filename;
  while(fcsv >> filename) {
    lines.push_back(filename);
  } 
 // if(argc == 4 && argv[3][0] == '1') {
 //   std::random_shuffle(lines.begin(), lines.end());
 // }
  LOG(INFO) << "A total of " << lines.size() << " csv files.";
  fcsv.close();

  // Setup the leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.error_if_exists = true;
  options.create_if_missing = true;
  options.write_buffer_size = 268435456;
  LOG(INFO) << "Opening leveldb " << argv[2];
  leveldb::Status status = leveldb::DB::Open(
      options, argv[2], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[2];
  
  //cv::namedWindow("debug");

  Datum datum; 
  int count = 0;
  const int maxKeyLength = 256;
  char key_cstr[maxKeyLength];
  leveldb::WriteBatch* batch = new leveldb::WriteBatch();
  for(int line_id = 0; line_id < lines.size(); line_id++) {
     // For each csv file. Read content into local variables
     // and then crop the images and write them to disk
     std::ifstream fboxes(lines[line_id].c_str());
     if(!fboxes.good()) {
        LOG(ERROR) << "Failed to open " << lines[line_id];
        continue;
     }
    
     LOG(INFO) << "#" << line_id;
     //read the file into fields
     string imgfilename;
     int numboxes;
     int boxdim;
     int imgset;
     int numlabels;
     int labeldim;
     vector<vector<int> > boxes;
     vector<vector<int> > labels;
     fboxes >> imgfilename >> imgset >> numboxes >> boxdim;
 //    if(imgset == 3 || imgset == 1) { continue; } // We are collecting val data only.
     for(int j = 0; j < numboxes; j++) {
         vector<int> temp(boxdim);
         fboxes >> temp[0] >> temp[1] >> temp[2] >> temp[3];
         boxes.push_back(temp);
     }
     fboxes >> numlabels >> labeldim;
     for(int j = 0; j < numlabels; j++) {
         vector<int> temp(labeldim);
         fboxes >> temp[0] >> temp[1] >> temp[2] >> temp[3]; // >> temp[4] >> temp[5] 
                         // >> temp[6] >> temp[7] >> temp[8] >> temp[9];
         
         labels.push_back(temp);
     }
     fboxes.close();
     
     LOG(INFO) << "Found " << numboxes << " boxes";

     // Change imgfilename to point to the right path on the server
     imgfilename.erase(imgfilename.begin(), 
            imgfilename.begin() + strlen("/nfs/onega_no_backups1/users/amahend1/"));
     imgfilename = "/data/VOCdevkit/" + imgfilename;
     
     // Iterate through the boxes and write things to leveldb
     cv::Mat img = cv::imread(imgfilename);
     if(img.empty()) {
        LOG(ERROR) << "Image read " << imgfilename << " failed";
        continue;
     }
     for(int boxno = 0; boxno < numboxes; boxno++) {
        /* if(labels[boxno][2] == 1 && labels[boxno][9] != 1) {  
           continue;  //Only the ground truth bounding boxes are the positives. The negatives come from selective search windows
         }*/
         /*if(labels[boxno][9] == 1 && imgset == 2) { 
           continue; // for now remove the ground truth labels from validation data.
         }*/
         cv::Mat imgcropped;
         cv::Rect roi(boxes[boxno][0], boxes[boxno][1], boxes[boxno][2]-boxes[boxno][0], boxes[boxno][3]-boxes[boxno][1]);
         cv::Mat(img, roi).copyTo(imgcropped);
         cv::Mat imgcroppedresized;
         cv::resize(imgcropped, imgcroppedresized, cv::Size(224, 224), 0, 0, cv::INTER_LANCZOS4);

         //cv::imshow("debug", imgcroppedresized);
         //cv::waitKey(50);
         // now convert this to datum 
         // note that labels[boxno][2] is not the right label because we have pruned out other boxes.
         convert_image_to_datum(imgcroppedresized, labels[boxno][2], &datum);

         snprintf(key_cstr, maxKeyLength, "%08d_%s_%08d", line_id, lines[line_id].c_str(), boxno);
         string value;
         datum.SerializeToString(&value);
         batch->Put(string(key_cstr), value);
         if (++count % 1000 == 0) {
           db->Write(leveldb::WriteOptions(), batch);
           LOG(ERROR) << "Processed " << count << " files.";
           delete batch;
           batch = new leveldb::WriteBatch();
         }
     }
  }
  // write the last batch
  if (count % 1000 != 0) {
    db->Write(leveldb::WriteOptions(), batch);
    LOG(ERROR) << "Processed " << count << " files.";
  }
  delete batch;
  delete db;
  return 0;
}

void convert_image_to_datum(cv::Mat& cv_img, int label, Datum* datum) {
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
}
