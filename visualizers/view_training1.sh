for i in '9000' '10000' '11000' '12000' '13000' '14000' '15000' '16000' '17000' '18000' '19000' '20000' '21000' '22000' '23000' '24000' '25000'
do
./build/visualizers/view_network.bin ../aravindhm_caffe/sparseconvautoencoder/net_deploy_manykernels.txt /snapshots/sparseconvautoencoder/caffe_sparseconvautoencoder_train_lr05_step1000_gamma9_momentum9_wdecay0005_verylargeeuclideanloss_lowl1sparse_manykernels_stride4_iter_$i sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4/iter$i/
mkdir -p /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4/
mkdir -p /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4/iter$i/
mv *.png /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4/iter$i/
mv index.html /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4/iter$i/
done
