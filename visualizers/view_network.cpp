// Copyright 2014 Aravindh Mahendran
//
// This program takes the netprototxt and pretrained net as input
// and generates visualizations for all the layers.
// It also dumps a html file index.html that can
// then be used to see these in the browser
// htmlprefix tells the prefix when define the src attribute of the img tag in index.html
//
// Usage:
//    view_network netprototxt trained_net htmlprefix
//


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
    cerr << "Usage: view_features_for_image netprototxt trained_net htmlprefix" << endl;
    return -1;
  }

  // Read the paramters and network from file
  NetParameter net_param;
  NetParameter trained_net_param;
  ReadProtoFromTextFile(argv[1], &net_param);
  ReadProtoFromBinaryFile(argv[2], &trained_net_param);

  // Load the network
  shared_ptr<Net<float> > caffe_net(new Net<float>(net_param));
  caffe_net->CopyTrainedLayersFrom(trained_net_param);

  // Load the layers in the network
  vector<shared_ptr<Layer<float> > > net_layers = caffe_net -> layers();

  fstream fout;
  fout.open("index.html", ios::out);
  if(!fout.good()) {
    cerr << "Error opening index.html" << endl;
    return -1;
  }
  fout << "<html><head><title>" << argv[1] << "</title></head>" << endl;
  fout << "<body>" << endl;
  // Iterate over the layers and generate color maps for the parameter blobs.
  for(int layerid = 0; layerid < net_layers.size(); layerid++) {
    fout << "<h4>" << net_layers[layerid]->layer_param().name() << "</h4> <hr/>" << endl;
    for(int blobid = 0; blobid < net_layers[layerid]->blobs().size(); blobid++) {
      fout << "<div style=\"clear:both\"> <hr/>" << endl;
      vector<cv::Mat> color_maps = Blob2ColorMap(net_layers[layerid]->blobs()[blobid]);
      for(int colormapid = 0; colormapid < color_maps.size(); colormapid++) {
        stringstream ss;
        ss << net_layers[layerid]->layer_param().name() << "_blob" << blobid << "_" << colormapid << ".png";
        cout << "Writing " << ss.str() << endl;
        cv::imwrite(ss.str(), color_maps[colormapid]);
        int width = max(net_layers[layerid]->blobs()[blobid]->width(), 50);
        int height = max(net_layers[layerid]->blobs()[blobid]->height(), 50);
        fout << "<div style=\"float:left;width:" << width + 5 << ";height:" << height + 5 << "\">" << endl;
        fout << "<img src=\"/" << argv[3] << ss.str() << "\" alt=\"" << ss.str() << "\" width=" 
             << width << " height=" 
             << height <<  " />" << endl;
        fout << "</div>" << endl;
      }
      fout << "</div>" << endl;
    }
  }
  fout << "</body></html>" << endl;
  fout.close();
  return 0;
}
