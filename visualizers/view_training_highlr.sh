for i in '9000' '10000' '11000' '12000'
do
./build/visualizers/view_network.bin ../aravindhm_caffe/sparseconvautoencoder/net_deploy_manykernels.txt /snapshots/sparseconvautoencoder/caffe_sparseconvautoencoder_train_lr05_step1000_gamma9_momentum9_wdecay0005_verylargeeuclideanloss_lowl1sparse_manykernels_stride4_highlr_iter_$i sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4highlr/iter$i/
mkdir -p /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4highlr/iter$i/
mv *.png /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4highlr/iter$i/
mv index.html /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4highlr/iter$i/
done
