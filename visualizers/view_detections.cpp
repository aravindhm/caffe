//Copyright 2014 Aravindh Mahendran

// view_detections view_detections imagefilename proposalsfilename svm_model net_proto_depoy trained_proto CPU/GPU

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <iomanip>

#include <omp.h>

#include <opencv2/opencv.hpp>

#include "caffe/caffe.hpp"

#include "/mnt/scratch/liblinear-1.94/linear.h"

#define BATCH_SIZE 256

using namespace std;
using namespace cv;
using namespace caffe;

int main(int argc, char* argv[]) {
  //load the network into memory
  if(argc < 6) {
     cerr << "Usage: view_detections imagefilename proposalsfilename svm_model net_proto_depoy trained_proto CPU/GPU" << endl;
     return -1;
  }
  // first parse the arguments. there's too many of them
  string imgfilename(argv[1]);
  string proposalsfilename(argv[2]);
  string svmmodelfilename(argv[3]);
  string netprotofilename(argv[4]);
  string trainedprotofilename(argv[5]);

  cudaSetDevice(0);
  Caffe::set_phase(Caffe::TEST);

  cout << argc << " " << argv[6] << endl;
  if (strcmp(argv[5], "GPU") == 0) {
    LOG(ERROR) << "Using GPU";
    Caffe::set_mode(Caffe::GPU);
  } else {
     LOG(ERROR) << "Using CPU";
     Caffe::set_mode(Caffe::CPU);
  }

  //Read the test image
  cv::Mat img = imread(imgfilename); // read the input image
  if(img.empty()) {
    cerr << "Error reading image file" << endl;
  }
  
  //Load the network
  NetParameter deploy_net_param;
  ReadProtoFromTextFile(netprotofilename, &deploy_net_param);
  Net<float> caffe_net(deploy_net_param);
  NetParameter trained_net_param;
  ReadProtoFromBinaryFile(trainedprotofilename, &trained_net_param);
  caffe_net.CopyTrainedLayersFrom(trained_net_param);
    
  LOG(INFO) << "Loaded network";

  // Open the file to store the features
  fstream fsvm;
  fsvm.open(".svm_data.txt", ios::out);
  if(!fsvm.good()) {
    cerr << "Error opening .svm_data.txt" << endl;
    return -1;
  }
  else {
    cout << "Recording features in .svm_data.txt" << endl;
  }
  
  //Create the blob for input
  int batch_size = deploy_net_param.input_dim(0);
  int height = deploy_net_param.input_dim(2);
  int width = deploy_net_param.input_dim(3);
  Blob<float>* input = new Blob<float>(batch_size, 3, height, width);

  // Read the proposal windows 
  fstream fin; //file stream instance for reading box proposal data.
  fin.open(proposalsfilename.c_str(), ios::in);
  if(!fin.good()) {
      cerr << "File open failed " << proposalsfilename << endl;
  }
  //read the file into fields
  int numboxes;
  int imgset;
  int junk;
  int numlabels;
  vector<vector<int> > boxes;
  vector<vector<int> > labels;
  fin >> imgfilename >> imgset >> numboxes >> junk;
  for(int i = 0; i < numboxes; i++) {
    vector<int> temp(4);
    fin >> temp[0] >> temp[1] >> temp[2] >> temp[3];
    boxes.push_back(temp);
  }
  fin >> numlabels >> junk;
  for(int i = 0; i < numlabels; i++) {
    vector<int> temp(5);
    fin >> temp[0] >> temp[1] >> temp[2] >> temp[3] >> temp[4];
    labels.push_back(temp);
  }

  fin.close();

  //iterate over the boxes and compute features and write to file for svm prediction
  for(int boxno = 0; boxno < numboxes; boxno++) {
    //cout << "Boxno: " << boxno << endl;
    //crop the image
    Mat imgcropped;
    Rect roi(boxes[boxno][0], boxes[boxno][1], boxes[boxno][2]-boxes[boxno][0], boxes[boxno][3]-boxes[boxno][1]);
    Mat(img, roi).copyTo(imgcropped);
    Mat imgcroppedresized;
    resize(imgcropped, imgcroppedresized, Size(100, 100), 0, 0, INTER_LANCZOS4);

    //copy it into the input blob and call net-> forward()
    float* input_data = input->mutable_cpu_data();
    for(int c = 0; c < 3; c++) {
      for(int h = 0; h < height; h++) {
        for(int w = 0; w < width; w++) {
          input_data[input->offset(0,c,h,w)] = imgcroppedresized.at<Vec3b>(h,w)[c]+0.0f;
        }
      }
    }
    vector<Blob<float>*> blob_input_vec;
    blob_input_vec.push_back(input);
    const vector<Blob<float>*>& result = caffe_net.Forward(blob_input_vec);
    int count = result[0]->count();
    const float* result_data = result[0]->cpu_data();

    if(labels[boxno][2] == 1) fsvm << "+1";
    else fsvm << "-1";
    for(int j = 0; j < count; j++) {
      if(result_data[j] != 0) {
        fsvm << " " << j+1 << ":" << result_data[j];
      }
    }
    fsvm << endl;
  }
  fsvm.close();

  stringstream ss; 
  ss << "/mnt/scratch/liblinear-1.94/predict .svm_data.txt " << svmmodelfilename << " .svm_output.txt";

  fstream fsvmoutput;
  fsvmoutput.open(".svm_output.txt", ios::in);
  vector<int> prediction(numboxes);
  for(int boxno = 0; boxno < numboxes; boxno++) {
    fsvmoutput >> prediction[boxno];
  }
  return 0;
}
