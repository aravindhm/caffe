for i in '3000' '4000' '5000' '6000' '7000' '8000' '9000' '10000'
do
./build/visualizers/view_network.bin ../aravindhm_caffe/sparseconvautoencoder/net_deploy_manykernels.txt /snapshots/sparseconvautoencoder/caffe_sparseconvautoencoder_train_lr05_step1000_gamma9_momentum9_wdecay0005_verylargeeuclideanloss_lowl1sparse_manykernels_stride4_highlr_relu_iter_$i sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4highlrrelu/iter$i/
mkdir -p /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4highlrrelu/iter$i/
mv *.png /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4highlrrelu/iter$i/
mv index.html /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4highlrrelu/iter$i/
done
