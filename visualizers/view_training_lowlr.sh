for i in '3000' '4000' '5000'
do
./build/visualizers/view_network.bin ../aravindhm_caffe/sparseconvautoencoder/net_deploy_manykernels.txt /snapshots/sparseconvautoencoder/caffe_sparseconvautoencoder_train_lr05_step1000_gamma9_momentum9_wdecay0005_verylargeeuclideanloss_lowl1sparse_manykernels_stride4_lowlr_iter_$i sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4lowlr/iter$i/
mkdir -p /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4lowlr/iter$i/
mv *.png /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4lowlr/iter$i/
mv index.html /opt/www/sparseautoencoder/lr05verylargelosslowl1sparsemanykernelsstride4lowlr/iter$i/
done
