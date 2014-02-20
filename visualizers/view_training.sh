for i in '9000' '10000' '11000' '12000' '13000' '14000' '15000' '16000'
do
./build/visualizers/view_network.bin ../aravindhm_caffe/sparseconvautoencoder/net_deploy.txt /data/kitti/caffe_sparseconvautoencoder_train_lr001_iter_$i sparseautoencoder/lr001/iter$i/
mkdir -p /var/www/sparseautoencoder/lr001/iter$i/
mv *.png /var/www/sparseautoencoder/lr001/iter$i/
mv index.html /var/www/sparseautoencoder/lr001/iter$i/
done
