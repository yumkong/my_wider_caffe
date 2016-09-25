function trained_model_compare()

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

opts.net_def_file = '..\models\rpn_prototxts\vgg_16layers_conv3_1\test_widerface.prototxt';
opts.net_file = '.\output\rpn_cachedir\faster_rcnn_WIDERFACE_vgg_16layers_stage1_rpn_2\WIDERFACE_train\final';
opts.test_image_dir = '.\test_image';

cache_dir = fullfile(pwd, 'output', 'test_one_image_cachedir');

mkdir_if_missing(cache_dir);
caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
caffe.init_log(caffe_log_file_base);
caffe_net = caffe.Net(opts.net_def_file, 'test');
caffe_net.copy_from(opts.net_file);

net = load('matconvnet_model\net-epoch-30.mat');
caffe_layer_pool = {'conv1_1','conv1_2', 'conv2_1', 'conv2_2','conv3_1','conv3_2','conv3_3','conv4_1',...
                    'conv4_2', 'conv4_3', 'conv5_1','conv5_2','conv5_3','conv_proposal1',...
                    'proposal_cls_score', 'proposal_bbox_pred'};
net = net.net;
for k = 1:16
    layer_name = caffe_layer_pool{k};
    %compute_param_diff(caffe_net, net.net, layer_name, 2*k - 1, 2*k);
    net.params(2*k - 1).value = caffe_net.params(layer_name,1).get_data();
    net.params(2*k).value = caffe_net.params(layer_name,2).get_data();
end

save('new-net-epoch-30.mat', 'net');

function compute_param_diff(caffe_net, mat_net, caffe_layer_name, mat_w_idx, mat_b_idx)
    caffe_w = caffe_net.params(caffe_layer_name,1).get_data();
    caffe_b = caffe_net.params(caffe_layer_name,2).get_data();
    matcov_w = mat_net.params(mat_w_idx).value;
    if mat_w_idx == 1
        matcov_w = matcov_w(:,:,[3,2,1],:);
    end
    matcov_w = permute(matcov_w, [2,1,3,4]);
    matcov_b = mat_net.params(mat_b_idx).value;
    diff_w = abs(matcov_w - caffe_w);
    res_w = sum(diff_w(:));
    diff_b = abs(matcov_b - caffe_b);
    res_b = sum(diff_b(:));
    fprintf('%s: res_w = %.4f, res_b = %.4f\n', caffe_layer_name, res_w, res_b);