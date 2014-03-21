import caffe;
import numpy as np;
from caffe.proto import caffe_pb2;

#Read the flow from file
flow = np.genfromtxt('/data/kitti/flow_sequences_rbig/00/image_2/flow/000000_siftflow.csv', delimiter=',');
flow = flow[np.newaxis, np.newaxis, :, :];
flow = np.ascontiguousarray(flow[:, :, 100:200, 500:600].astype(np.float32))
print flow.shape

input_blobs = [flow];
output_blobs = [np.empty((1, 1, 5, 5), dtype=np.float32)];

caffenet = caffe.CaffeNet('/code/caffe_flowprediction/flowprediction/flow_deploy.prototxt', '/snapshots/flowprediction/flowpredictionnet_trainlr0001_step100000_gamma5_momentum9_wdecay0005_imagenetprefilledconv1conv2_moredata_iter_215000');
caffenet.set_phase_test()
caffenet.set_mode_cpu()
caffenet.Forward(input_blobs, output_blobs);
output_flow = output_blobs[0].reshape((5,5));
print output_flow


print "good day"

np.savetxt("flow.csv", flow.reshape((100, 100)), delimiter=",", fmt="%.3f")
