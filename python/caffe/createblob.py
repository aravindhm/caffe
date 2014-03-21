import caffe;
import caffe.convert;
import numpy as np;

arr = np.random.rand(1,1,60,60).astype(np.float32);
print arr;
blob = caffe.convert.array_to_blobproto(arr);
strblob = blob.SerializeToString();
open('testwhiteningblob.binaryproto', 'wb').write(strblob);
