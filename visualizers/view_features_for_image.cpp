// Copyright 2014 Aravindh Mahendran
//
// This program takes an image filename as argument
// and generates the color map of the features in all the layers
// 
// Usage:
//   view_features_for_image imagefilename netprototxt trained_net htmlprefix

#include <opencv2/opencv.hpp>

#include "caffe/util/io.hpp"
#include "caffe/net.hpp"
#include "caffe/common.hpp"
#include "caffe/blob.hpp"

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>

using namespace std;
using namespace boost;
using namespace caffe;

int main(int argc, char* argv[]) {
  if(argc < 4) {
    cerr << "Usage: view_features_for_image imagefilename netprototxt trained_net htmlprefix" << endl;
    return -1;
  }

  // First read the image into an vector of blobs
  cv::Mat img = cv::imread(argv[1]);
  if(img.empty()) {
    cerr << "Error reading image " << argv[1] << endl;
    return -1;
  }
  shared_ptr<Blob<float> > input_blob = CvMatToBlob(img);
  vector<Blob<float>* > input_vec;
  input_vec.push_back(input_blob.get());

  // Read the paramters and network from file
  NetParameter net_param;
  NetParameter trained_net_param;
  ReadProtoFromTextFile(argv[2], &net_param);
  ReadProtoFromBinaryFile(argv[3], &trained_net_param);

  // Forward pass through the network and extract the features
  shared_ptr<Net<float> > caffe_net(new Net<float>(net_param));
  caffe_net->CopyTrainedLayersFrom(trained_net_param);

  cout << "Performing Forward" << endl;
  caffe_net->Forward(input_vec);
  cout << "Forward done" << endl;
  
  const vector<string>& blob_names = caffe_net->blob_names();
  const vector<shared_ptr<Blob<float> > >& blobs = caffe_net->blobs();

  //open index.html for writing.
  fstream fout;
  fout.open("index.html", ios::out);
  if(!fout.good()) {
    cerr << "Error opening index.html" << endl;
    return -1;
  }
  
  //write the html header
  fout << "<html><head><title>" << argv[1] << "</title></head>" << endl;
  fout << "<body>" << endl;

  for (int blobid = 0; blobid < caffe_net->blobs().size(); ++blobid) {
    fout << "<div style=\"clear:both\"><h4>" << blob_names[blobid] << "</h4></div>" << endl;
    // Serialize blob
    vector<shared_ptr<cv::Mat> > color_maps = Blob2ColorMap(blobs[blobid]);
    for(int colormapid = 0; colormapid < color_maps.size(); colormapid++) {
      stringstream ss;
      ss << blob_names[blobid] << "_" << colormapid << ".png";
      cout << "Writing " << ss.str() << endl;
      int new_width = min(color_maps[colormapid]->cols, 640);
      int new_height = (color_maps[colormapid]->rows*1.0/color_maps[colormapid]->cols) * new_width;
      cout << "Resizing to " << new_width << "x" << new_height << endl;
      cv::Mat img_to_be_written;
      cv::resize(*(color_maps[colormapid].get()), img_to_be_written, cv::Size(new_width, new_height));
      cv::imwrite(ss.str(), img_to_be_written);

      int width = img_to_be_written.cols;
      int height = img_to_be_written.rows;
      fout << "<div style=\"float:left;width:" << width + 5 << ";height:" << height + 5 << "\">" << endl;
      fout << "<img src=\"/" << argv[4] << ss.str() << "\" alt=\"" << ss.str() << "\" width=" 
	      << width << " height=" 
	      << height <<  " />" << endl;
      fout << "</div>" << endl;
    }
    fout << "</div>" << endl;
  }
  fout << "</body></html>" << endl;
  fout.close();

  return 0;
}
