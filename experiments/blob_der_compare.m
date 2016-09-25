function blob_der_compare(caffe_net, caffe_param_name, mat_w, mat_b)

% aa = caffe_net.blobs(caffe_blob_name).get_diff();
% bb = mat_blob;
% cc = abs(aa - bb);

caffe_w = caffe_net.params(caffe_param_name,1).get_diff();
caffe_b = caffe_net.params(caffe_param_name,2).get_diff();
w_diff = abs(caffe_w - mat_w);
b_diff = abs(caffe_b - mat_b);
format long
fprintf('%s derivative w_diff: %.10f, b_diff: %.10f\n', caffe_param_name, sum(w_diff(:)), sum(b_diff(:)));