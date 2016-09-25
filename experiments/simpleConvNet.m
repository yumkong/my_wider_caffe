clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
% -------------------- CONFIG --------------------
%opts.caffe_version          = 'simple_conv_net';
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

caffe.set_mode_cpu();

net = caffe.Net('conv.prototxt', 'test');

im = single(imread('../004545.jpg'));
im_mean = load(fullfile('..', 'models', 'pre_trained_models', 'vgg_16layers', 'mean_image'));
im_mean = im_mean.image_mean;

im_blob = bsxfun(@minus, im, im_mean);
im_blob = imresize(im_blob, [224 224]);

%net.blobs('data').set_data({im_blob});
%net.reshape_as_input({im_blob});
%net.set_input_data({im_blob});
net.forward({im_blob});

conv_output = net.blobs('conv').get_data();
w = net.params('conv',1).get_data();
b = net.params('conv',2).get_data();

save('simple.mat','im_blob','conv_output', 'w', 'b');