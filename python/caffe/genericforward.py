import caffe;
import caffe.convert;
import cv2;
import numpy as np;
from caffe.proto import caffe_pb2;

blob = caffe_pb2.BlobProto();
data = open('/data/kitti/kitti_t1_mean.binaryproto').read();
blob.ParseFromString(data);
mean = caffe.convert.blobproto_to_array(blob);

img = cv2.imread('/data/kitti/sequences/00/image_2/000000.png');
img = img.astype(np.float32);
image = np.ascontiguousarray(img.swapaxes(1, 2).swapaxes(0, 1)[np.newaxis, :,:,:]);
image = image - mean;
image = image[:, :, 100:200, 500:600].astype(np.float32);
cv2.imwrite("img.png", img[100:200, 500:600, :].astype(np.uint8));

input_blobs = [image];
output_blobs = [np.empty((1, 25, 1, 1), dtype=np.float32)];

caffenet = caffe.CaffeNet('/code/caffe_flowprediction/flowprediction/imagenet_deploy.prototxt', '/snapshots/flowprediction/flowpredictionnet_trainlr0001_step100000_gamma5_momentum9_wdecay0005_imagenetprefilledconv1conv2_moredata_iter_215000');
caffenet.Forward(input_blobs, output_blobs);
output_flow = output_blobs[0].reshape((5,5));
print output_flow

