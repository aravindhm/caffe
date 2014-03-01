for i in '17000'
do
./build/visualizers/view_network.bin ../aravindhm_caffe/sparseconvautoencoder/net_deploy_manykernels.txt /snapshots/sparseconvautoencoder/caffe_sparseconvautoencoder_train_lr05_step1000_gamma9_momentum9_wdecay0005_largeeuclideanloss_lowl1sparse_manykernels_iter_$i sparseautoencoder/lr05largelosslowl1sparsemanykernels/iter$i/
mkdir -p /var/www/sparseautoencoder/lr05largelosslowl1sparsemanykernels/
mkdir -p /var/www/sparseautoencoder/lr05largelosslowl1sparsemanykernels/iter$i/
mv *.png /var/www/sparseautoencoder/lr05largelosslowl1sparsemanykernels/iter$i/
mv index.html /var/www/sparseautoencoder/lr05largelosslowl1sparsemanykernels/iter$i/
done
