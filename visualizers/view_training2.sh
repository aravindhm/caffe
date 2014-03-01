for i in '1000' '3000' '5000' '7000' '9000' '11000' '13000' '15000'
do
./build/visualizers/view_network.bin ../aravindhm_caffe/sparseconvautoencoder/net_deploy_manykernels.txt /snapshots/sparseconvautoencoder/caffe_sparseconvautoencoder_train_lr05_step1000_gamma9_momentum9_wdecay0005_largeeuclideanloss_lowl1sparse_manykernels_stride2_iter_$i sparseautoencoder/lr05largelosslowl1sparsemanykernelsstride2/iter$i/
mkdir -p /opt/www/sparseautoencoder/lr05largelosslowl1sparsemanykernelsstride2/
mkdir -p /opt/www/sparseautoencoder/lr05largelosslowl1sparsemanykernelsstride2/iter$i/
mv *.png /opt/www/sparseautoencoder/lr05largelosslowl1sparsemanykernelsstride2/iter$i/
mv index.html /opt/www/sparseautoencoder/lr05largelosslowl1sparsemanykernelsstride2/iter$i/
done
