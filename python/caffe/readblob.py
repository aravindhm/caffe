import caffe;
import caffe.convert;
import numpy as np;
from caffe.proto import caffe_pb2;

strblob = open('testwhiteningblob.binaryproto', 'rb').read();
blob = caffe_pb2.BlobProto();
blob.ParseFromString(strblob);
arr = caffe.convert.blobproto_to_array(blob);
print arr.shape
np.savetxt('temp.txt', arr[0,0,:,:], fmt='%.3f', delimiter=',');
