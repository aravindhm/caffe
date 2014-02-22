for i in '170000' '160000' '150000' '140000' '130000'
do
./build/visualizers/view_network.bin ../aravindhm_caffe/sparseconvautoencoder/net_deploy.txt /data/experiments/sparseconvautoencoder/caffe_sparseconvautoencoder_train_lr01_iter_$i sparseautoencoder/lr01/iter$i/
mkdir -p /var/www/sparseautoencoder/lr01/iter$i/
mv *.png /var/www/sparseautoencoder/lr01/iter$i/
mv index.html /var/www/sparseautoencoder/lr01/iter$i/
done
