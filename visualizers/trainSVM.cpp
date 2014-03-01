//Copyright 2014 Aravindh Mahendran

// trainSVM net_proto_depoy trained_proto CPU/GPU data_filename

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

//Train: 5006804 Val: 5045863 Test: 9941962

#define NUM_DATA 117300
#define NUMFILES 50

/*
Train: 1009184
Val: 971680
Test: 2010255
*/

int getNumBoxesTotal(vector<int>& car_train) {
    int result = 0;
    fstream fin; //file stream instance for reading box proposal data.
    for(int fileno = 0; fileno < NUMFILES; fileno++) {
      //compute the filename and open the file
      stringstream ss;
      ss << "/mnt/scratch/csv/VOC2007_proposalswithlabels_" << setw(6) << setfill('0') << car_train[fileno] << ".csv";
      fin.open(ss.str().c_str(), ios::in);
      if(!fin.good()) {
          cerr << "File open failed " << ss.str() << endl;
      }

      //read the file into fields
      string imgfilename;
      int numboxes;
      int imgset;
      int junk;
      fin >> imgfilename >> imgset >> numboxes >> junk;
      result = result + BATCH_SIZE; //(numboxes/BATCH_SIZE)*BATCH_SIZE;
      fin.close();
    }
    return result;
}

int main(int argc, char* argv[]) {

    omp_set_num_threads(3);
    omp_set_dynamic(0);

    //load the network into memory
    if(argc < 5) {
       cerr << "Usage: trainSVM net_proto_depoy trained_proto [CPU/GPU]" << endl;
       return -1;
    }
    LOG(INFO) << "food";
    cudaSetDevice(0);
    Caffe::set_phase(Caffe::TEST);

    cout << argc << " " << argv[3] << endl;
    if (strcmp(argv[3], "GPU") == 0) {
      LOG(ERROR) << "Using GPU";
      Caffe::set_mode(Caffe::GPU);
    } else {
      LOG(ERROR) << "Using CPU";
      Caffe::set_mode(Caffe::CPU);
    }

    fstream fsvm;
    fsvm.open(argv[4], ios::out);
    if(!fsvm.good()) {
      cerr << "Error opening svm_data" << endl;
      return -1;
    }
    else {
      cout << "Recording features in " << argv[4] << endl;
    }

    // Read what images are training and validation. We cannot use  test data for model refinement.
    vector<int> car_train;
    int val;
    int label;
    fstream ftrainval;
    ftrainval.open("./car_val.txt", ios::in);
    while(!ftrainval.eof()) {
      ftrainval >> val >> label;
      car_train.push_back(val);
    }
    ftrainval.close(); 
    LOG(INFO) << "Car train has " << car_train.size() << " instances";
/*
    struct problem prob;
    prob.bias = 1.0;
    prob.n = 4097;
    prob.l = getNumBoxesTotal(car_train); // Number of data points;
    LOG(INFO) << "Got " << prob.l << " data points" << endl;
    prob.y = (double*)malloc(prob.l*sizeof(double));
    prob.x = (struct feature_node**)malloc(prob.l*sizeof(struct feature_node*));
    for(int i = 0; i < prob.l; i++) {
       prob.x[i] = (struct feature_node*)malloc(prob.n*sizeof(struct feature_node));
    }      
    LOG(INFO) << "Problem memory allocation complete";
    struct parameter svm_param;
    svm_param.solver_type = L2R_L2LOSS_SVC_DUAL;
    svm_param.C = 1.0;
*/

    NetParameter deploy_net_param;
    ReadProtoFromTextFile(argv[1], &deploy_net_param);
    Net<float> caffe_net(deploy_net_param);
    NetParameter trained_net_param;
    ReadProtoFromBinaryFile(argv[2], &trained_net_param);
    caffe_net.CopyTrainedLayersFrom(trained_net_param);
    
    LOG(INFO) << "Loaded network";
    
    int batch_size = deploy_net_param.input_dim(0);
    int height = deploy_net_param.input_dim(2);
    int width = deploy_net_param.input_dim(3);
    Blob<float>* input = new Blob<float>(batch_size, 3, height, width);

    int counter = 0; //keep track of the data point number;

    fstream fin; //file stream instance for reading box proposal data.
    for(int fileno = 0; fileno < NUMFILES; fileno++) {
      //compute the filename and open the file
      stringstream ss;
      ss << "/mnt/scratch/csv/VOC2007_proposalswithlabels_" << setw(6) << setfill('0') << car_train[fileno] << ".csv";
      fin.open(ss.str().c_str(), ios::in);
      if(!fin.good()) {
          cerr << "File open failed " << ss.str() << endl;
      }

      //read the file into fields
      string imgfilename;
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

      // fix the filename so that it points to the correct location on server
      int posvoc2007 = imgfilename.find("VOC2007");
      imgfilename = "/data/VOCdevkit/" + imgfilename.substr(posvoc2007);
      // after reading the information, process each box to compute features and write them to disk
      
      cout << fileno << " " << imgfilename << endl; 
      Mat img = imread(imgfilename); 
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
/*
         for(int j = 0; j < count; j++) {
           prob.x[counter][j].index = j+1; 
           prob.x[counter][j].value = result_data[j]; 
           prob.y[counter] = (int)round((labels[boxno][2]-0.5)*2); 
         }
         prob.x[counter][count].index = -1;
         counter++;*/
         if(labels[boxno][2] == 1) fsvm << "+1";
         else fsvm << "-1";
         for(int j = 0; j < count; j++) {
           if(result_data[j] != 0) {
             fsvm << " " << j+1 << ":" << result_data[j];
           }
         }
         fsvm << endl;
      }
    }
    /*for(int i = 0; i < prob.l; i++) 
    { cout << prob.y[i];
      for(int j = 0; j < prob.n; j++) {
        cout << " " << prob.x[i][j].value;
      }
      cout << endl;
    }
    return 0;
    double *target; // = (double*)malloc(prob.l*sizeof(double));
    int errors = 0;
    cout << "Running cross validation" << endl;
    cross_validation(&prob, &svm_param, 10, target);
    cout << "Cross validation done" << endl;
    for(int i = 0; i < prob.l; i++) {
      errors += (target[i] != prob.y[i]);
    }
    cout << "Accuracy: " << (prob.l - errors)/prob.l << endl;*/
    fsvm.close(); 
    /*fstream fout;
    fout.open("features1to2000_feats.bin", ios::out | ios::binary);
    fout.write(reinterpret_cast<char*>(&total_data), sizeof(total_data));
    fout.close();*/
    return 0;
}
