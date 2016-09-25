clear
clc
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

opts.solver_def_file        = fullfile('..', 'models', 'rpn_prototxts', 'vgg_16layers_conv3_1', 'solver_60k80k_widerface.prototxt');
opts.net_file               = fullfile('..', 'models', 'pre_trained_models', 'vgg_16layers', 'vgg16.caffemodel');

cache_dir = fullfile(pwd, 'cache_net_visualize');
mkdir_if_missing(cache_dir);
caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
caffe.init_log(caffe_log_file_base);

%load the training net
caffe_solver = caffe.Solver(opts.solver_def_file);
caffe_solver.net.copy_from(opts.net_file);

layer_num = numel(caffe_solver.net.layer_names);
layer_paramvalue_map = containers.Map();
for k = 1:layer_num
    layer_name = caffe_solver.net.layer_names{k};
    if strcmp(layer_name(1:4), 'conv') && length(layer_name) <= 10
        layer_paramvalue_map([layer_name '_w']) = caffe_solver.net.params(layer_name,1).get_data();
        layer_paramvalue_map([layer_name '_b']) = caffe_solver.net.params(layer_name,2).get_data();
    end
end
% save result
%save('layer_paramvalue_map.mat', 'layer_paramvalue_map');

%load the test net
opts.net_def_file = '..\models\rpn_prototxts\vgg_16layers_conv3_1\test_widerface.prototxt';
opts.net_file_finetuned = '.\output\rpn_cachedir\faster_rcnn_WIDERFACE_vgg_16layers_stage1_rpn_2\WIDERFACE_train\final';
caffe_net = caffe.Net(opts.net_def_file, 'test');
caffe_net.copy_from(opts.net_file_finetuned);

test_layer_num = numel(caffe_net.layer_names);
layer_paramvalue_map_test = containers.Map();
for k = 1:test_layer_num
    layer_name = caffe_net.layer_names{k};
    if strcmp(layer_name(1:4), 'conv') && length(layer_name) <= 10
        layer_paramvalue_map_test([layer_name '_w']) = caffe_net.params(layer_name,1).get_data();
        layer_paramvalue_map_test([layer_name '_b']) = caffe_net.params(layer_name,2).get_data();
    end
end
% save result
%save('layer_paramvalue_map_test.mat', 'layer_paramvalue_map_test');

% compare the raw params and fine-tuned params
fprintf('\n Caffe: |finetuned param - raw param|\n');
key_names = layer_paramvalue_map.keys;
for j = 1:layer_paramvalue_map.length
    abs_diff = abs(layer_paramvalue_map_test(key_names{j}) - layer_paramvalue_map(key_names{j}));
    fprintf('%s param difference: %.5f\n', key_names{j}, sum(abs_diff(:)));
end

% load matconvnet's model for comparison
addpath('matconvnet_model\');
addpath('matconvnet_model\help_func\');
addpath('matconvnet_model\simplenn\');

load('matconvnet_model\net-epoch-7.mat');   % fine-tuned nets

net1 = load('matconvnet_model\imagenet-vgg-verydeep-16.mat'); %original vgg
net1 = vl_simplenn_tidy(net1);
net1 = dagnn.DagNN.fromSimpleNN(net1, 'canonicalNames', true); 
fprintf('\n MatConvNet: |finetuned param - raw param|\n');
for i = 1:26
    abs_diff = abs(net.params(i).value - net1.params(i).value);
    fprintf('param name: %s, square diff: %.3f\n', net.params(i).name, sum(abs_diff(:)));
end

% compare the raw params between caffe and matconvnet (fit matconvnet params to caffe style)
fprintf('\n |matconvnet raw param - caffe raw param|\n');
for j = 1:layer_paramvalue_map.length
    % matconvnet order:  conv1_1_w, conv1_1_b, conv1_2_w, conv1_2_b ...
    % caffe      order:  conv1_1_b, conv1_1_w, conv1_2_b, conv1_2_w ...
    % get matconvnet param : first should switch w and b
    if mod(j,2) == 1  %odd index
        mat_param = net1.params(j+1).value;
    else  %even index
        mat_param = net1.params(j-1).value;  
        assert(length(size(mat_param)) == 4);  % h x w x ch x num
        mat_param = permute(mat_param, [2,1,3,4]);
    end
    
    % if conv1_1_w, rgb --> brg
    if strcmp(key_names{j}(1:9), 'conv1_1_w')
        mat_param = mat_param(:, :, [3, 2, 1], :); % from rgb to brg
    end
    
    abs_diff = abs(mat_param - layer_paramvalue_map(key_names{j}));
    fprintf('%s param difference: %.5f\n', key_names{j}, sum(abs_diff(:)));
end

% compare the finetuned params between caffe and matconvnet (fit matconvnet params to caffe style)
fprintf('\n |matconvnet finetuned param - caffe finetuned param|\n');
for j = 1:layer_paramvalue_map_test.length
    % matconvnet order:  conv1_1_w, conv1_1_b, conv1_2_w, conv1_2_b ...
    % caffe      order:  conv1_1_b, conv1_1_w, conv1_2_b, conv1_2_w ...
    % get matconvnet param : first should switch w and b
    if mod(j,2) == 1  %odd index
        mat_param = net.params(j+1).value;
    else  %even index
        mat_param = net.params(j-1).value;  
        assert(length(size(mat_param)) == 4);  % h x w x ch x num
        mat_param = permute(mat_param, [2,1,3,4]);
    end
    
    % if conv1_1_w, rgb --> brg
    if strcmp(key_names{j}(1:9), 'conv1_1_w')
        mat_param = mat_param(:, :, [3, 2, 1], :); % from rgb to brg
    end
    
    abs_diff = abs(mat_param - layer_paramvalue_map_test(key_names{j}));
    fprintf('%s param difference: %.5f\n', key_names{j}, sum(abs_diff(:)));
end
