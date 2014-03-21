import leveldb;
from caffe.proto import caffe_pb2

import cv2;
import numpy as np;

db = leveldb.LevelDB('/leveldb/pascal/224x224toy');

for i in xrange(1670,1677):
  key = '00000000_/data/csvgtappendedtrainval/VOC2007_proposalswithlabels_000012.csv_{0:08d}'.format(i)
  print key
  value = db.Get(key)
  datum = caffe_pb2.Datum();
  datum.ParseFromString(value);
  print datum.channels
  print datum.height
  print datum.width
  img = np.zeros((datum.height,datum.width,3), np.uint8)
  for c in xrange(0,3):
     for h in xrange(0,224):
        for w in xrange(0,224):
           img[h,w,c] = ord(datum.data[(c*224 + h)*224 + w]);
  cv2.namedWindow("debug");
  cv2.imshow("debug", img);
  cv2.waitKey(0);
