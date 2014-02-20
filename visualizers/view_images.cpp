// Copyright 2014 Aravindh Mahendran
//
// This program simply reads a list of filenames and loads the corresponding images and displays them. 
// This helps test the corresponding io utility function
// Usage:
//    view_images fileslist.txt

#include <opencv2/opencv.hpp>

#include "caffe/util/io.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

using std::cout;
using std::endl;
using std::cerr;
using std::fstream;
using std::string;
using std::vector;

int main(int argc, char* argv[]) {
  if(argc < 2) {
     cerr << "Usage: view_images fileslist.txt" << endl;
     return -1;
  }
  
  fstream fin;
  fin.open(argv[1], std::ios::in);
  if(!fin.good()) {
    cerr << "Error opening " << argv[0] << endl;
    return -1;
  }

  vector<string> filenames;
  string filename;
  while(!fin.eof()) {
    fin >> filename;
    filenames.push_back(filename);
  }

  vector<boost::shared_ptr<cv::Mat> > images = caffe::ReadImagesToCvMat(filenames);

  cv::namedWindow("images", cv::WINDOW_AUTOSIZE);
  for(vector<boost::shared_ptr<cv::Mat> >::iterator it = images.begin(); it != images.end(); it++) {
      cv::imshow("images", *(it->get()));
      cv::waitKey(0);
  }
  
  return 0;
}
