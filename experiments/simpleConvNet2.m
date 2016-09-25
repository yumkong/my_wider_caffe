%function simpleConvNet2()
clc;
clear;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
% -------------------- CONFIG --------------------
opts.caffe_version          = 'caffe_faster_rcnn';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

caffe.set_mode_cpu();

%net = caffe.Net('conv_simple.prototxt', 'test');
cache_dir = fullfile(pwd, 'output', 'simple_conv_cachedir');
mkdir_if_missing(cache_dir);
caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
caffe.init_log(caffe_log_file_base);

caffe_solver = caffe.Solver('solver_simple.prototxt');
try
    param = load('conv_simple_param.mat');
    w = param.w;
    b = param.b;
catch
    w = caffe_solver.net.params('conv', 1).get_data();
    b = caffe_solver.net.params('conv', 2).get_data();
    save('conv_simple_param.mat', 'w', 'b');
end
% initialize conv parameters
caffe_solver.net.params('conv',1).set_data(w);
caffe_solver.net.params('conv',2).set_data(b);

im = single(reshape(1:27, 3,3,3,1));
label = 6;
euclidean_error_caffe = zeros(1000,1, 'single');
for i = 1:1000
%forward and backward
net_inputs = {im, label};
caffe_solver.net.reshape_as_input(net_inputs);
% one iter SGD update
caffe_solver.net.set_input_data(net_inputs);

caffe_solver.step(1);

% forward
%caffe_solver.net.forward({im, label});
conv_output = caffe_solver.net.blobs('conv').get_data();
loss_output = caffe_solver.net.blobs('loss').get_data();

% backward
%caffe_solver.net.backward_prefilled();
conv_diff = caffe_solver.net.blobs('conv').get_diff();
conv_diff_w = caffe_solver.net.params('conv',1).get_diff();
out_w = caffe_solver.net.params('conv',1).get_data();
out_b = caffe_solver.net.params('conv',2).get_data();

fprintf('Step %d: euclidean loss: %.10f\n', i, loss_output);
euclidean_error_caffe(i) = loss_output;
end

save('euclidean_error_caffe.mat','euclidean_error_caffe');