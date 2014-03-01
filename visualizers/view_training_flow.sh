for i in '1000' '2000' '3000' '4000' '5000' '6000' '7000' '8000' '40000'
do
./build/visualizers/view_network.bin ../caffe_flowprediction/flowprediction/imagenet_deploy.prototxt /snapshots/flowprediction/flowpredictionnet_trainlr0001_step100000_gamma5_momentum9_wdecay0005_imagenetprefilledconv1conv2_moredata_iter_$i flowprediction/imagenetprefilledconv1conv2/moredata/iter$i/
mkdir -p /opt/www/flowprediction/imagenetprefilledconv1conv2/moredata/iter$i/
mv *.png /opt/www/flowprediction/imagenetprefilledconv1conv2/moredata/iter$i/
mv index.html  /opt/www/flowprediction/imagenetprefilledconv1conv2/moredata/iter$i/
done
