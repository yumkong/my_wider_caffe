function test_one_image()

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

model.mean_image                                = fullfile('..', 'models', 'pre_trained_models', 'vgg_16layers', 'mean_image');
% Stride in input image pixels at the last conv layer
model.feat_stride                               = 16;
cache_base_proposal         = 'faster_rcnn_WIDERFACE_vgg_16layers';
model.stage1_rpn.cache_name = [cache_base_proposal, '_stage1_rpn'];
model.stage1_rpn.test_net_def_file              = fullfile('..', 'models', 'rpn_prototxts', 'vgg_16layers_conv3_1', 'test_widerface.prototxt');

conf_proposal               = proposal_config_widerface('image_means', model.mean_image, 'feat_stride', model.feat_stride);
% generate anchors and pre-calculate output size of rpn network 
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);

conf = conf_proposal;

cache_dir = fullfile(pwd, 'output', 'test_one_image_cachedir');

mkdir_if_missing(cache_dir);
caffe_log_file_base = fullfile(cache_dir, 'caffe_log');
caffe.init_log(caffe_log_file_base);
caffe_net = caffe.Net(opts.net_def_file, 'test');
caffe_net.copy_from(opts.net_file);

% init log
timestamp = datestr(datevec(now()), 'yyyymmdd_HHMMSS');
mkdir_if_missing(fullfile(cache_dir, 'log'));
log_file = fullfile(cache_dir, 'log', ['test_', timestamp, '.txt']);
diary(log_file);

% set random seed
prev_rng = seed_rand(conf.rng_seed);
caffe.set_random_seed(conf.rng_seed);

% set gpu/cpu
if conf.use_gpu
    caffe.set_mode_gpu();
else
    caffe.set_mode_cpu();
end

imgs = dir([opts.test_image_dir '\*.jpg']);
for i = 1:numel(imgs)
    th = tic;
    imname = imgs(i).name;
    im = imread([opts.test_image_dir '\' imname]);
    %[boxes, scores, abox_deltas{i}, aanchors{i}, ascores{i}] = detect_im_proposal(conf, caffe_net, im, i);
    detect_im_proposal(conf, caffe_net, im, i);

    fprintf(' time: %.3fs\n', toc(th));
end

diary off;
caffe.reset_all(); 
rng(prev_rng);
end

function detect_im_proposal(conf, caffe_net, im, im_idx)
%function [pred_boxes, scores, box_deltas_, anchors_, scores_] = detect_im_proposal(conf, caffe_net, im, im_idx)
% [pred_boxes, scores, box_deltas_, anchors_, scores_] = proposal_im_detect(conf, im, net_idx)
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------    
    use_original_size = true;
    
    if use_original_size
    % ------------ alter1: direct size
        im_blob = single(im);
        im_blob = bsxfun(@minus, im_blob, conf.image_means);
    else
        % ------------ alter2: re-size
        im = single(im);
        [im_blob, im_scales] = get_image_blob(conf, im);
        im_size = size(im);
        scaled_im_size = round(im_size * im_scales);
    end
    % permute data into caffe c++ memory, thus [num, channels, height, width]
    im_blob = im_blob(:, :, [3, 2, 1], :); % from rgb to brg
    im_blob = permute(im_blob, [2, 1, 3, 4]);
    im_blob = single(im_blob);

    net_inputs = {im_blob};

    % Reshape net's input blobs
    caffe_net.reshape_as_input(net_inputs);
    output_blobs = caffe_net.forward(net_inputs);

    % Apply bounding-box regression deltas
    box_deltas = output_blobs{1};
    featuremap_size = [size(box_deltas, 2), size(box_deltas, 1)];
    % permute from [width, height, channel] to [channel, height, width], where channel is the
        % fastest dimension
    box_deltas = permute(box_deltas, [3, 2, 1]);
    box_deltas = reshape(box_deltas, 4, [])';
    
    anchors = proposal_locate_anchors(conf, size(im), conf.test_scales, featuremap_size);
    pred_boxes = fast_rcnn_bbox_transform_inv(anchors, box_deltas);
      % scale back
    if ~use_original_size
        pred_boxes = bsxfun(@times, pred_boxes - 1, ...
        ([im_size(2), im_size(1), im_size(2), im_size(1)] - 1) ./ ([scaled_im_size(2), scaled_im_size(1), scaled_im_size(2), scaled_im_size(1)] - 1)) + 1;
    end
    pred_boxes = clip_boxes(pred_boxes, size(im, 2), size(im, 1));
    
    assert(conf.test_binary == false);
    % use softmax estimated probabilities
    scores = output_blobs{2}(:, :, end);
    scores = reshape(scores, size(output_blobs{1}, 1), size(output_blobs{1}, 2), []);
    % permute from [width, height, channel] to [channel, height, width], where channel is the
        % fastest dimension
    scores = permute(scores, [3, 2, 1]);
    scores = scores(:);
    
    %box_deltas_ = box_deltas;
    %anchors_ = anchors;
    %scores_ = scores;
    
    if conf.test_drop_boxes_runoff_image
        contained_in_image = is_contain_in_image(anchors, round(size(im) * im_scales));
        pred_boxes = pred_boxes(contained_in_image, :);
        scores = scores(contained_in_image, :);
    end
    
    % drop too small boxes
    [pred_boxes, scores] = filter_boxes(conf.test_min_box_size-3, pred_boxes, scores);
    
    % sort
    [scores, scores_ind] = sort(scores, 'descend');
    pred_boxes = pred_boxes(scores_ind, :);
    
    %image(single(im)/255); 
    imshow(single(im)/255); 
    axis image;
    axis off;
    set(gcf, 'Color', 'white');
    endNum = sum(scores >= 0.9);
    for j = 1:endNum  % can be changed to any positive number to show different #proposals
        bbox = pred_boxes(j,:);
        rect = [bbox(:, 1), bbox(:, 2), bbox(:, 3)-bbox(:,1)+1, bbox(:,4)-bbox(2)+1];
        rectangle('Position', rect, 'LineWidth', 1, 'EdgeColor', [0 1 0]);
    end
    saveName = sprintf('test_result\\img_%d_score_0.9_resize_0914',im_idx);
    export_fig(saveName, '-png', '-a1', '-native');
    fprintf('image %d saved.\n', im_idx);
end

function [blob, im_scales] = get_image_blob(conf, im)
    if length(conf.test_scales) == 1
        [blob, im_scales] = prep_im_for_blob(im, conf.image_means, conf.test_scales, conf.test_max_size);
    else
        [ims, im_scales] = arrayfun(@(x) prep_im_for_blob(im, conf.image_means, x, conf.test_max_size), conf.test_scales, 'UniformOutput', false);
        im_scales = cell2mat(im_scales);
        blob = im_list_to_blob(ims);    
    end
end

function [boxes, scores] = filter_boxes(min_box_size, boxes, scores)
    widths = boxes(:, 3) - boxes(:, 1) + 1;
    heights = boxes(:, 4) - boxes(:, 2) + 1;
    
    valid_ind = widths >= min_box_size & heights >= min_box_size;
    boxes = boxes(valid_ind, :);
    scores = scores(valid_ind, :);
end

function boxes = clip_boxes(boxes, im_width, im_height)
    % x1 >= 1 & <= im_width
    boxes(:, 1:4:end) = max(min(boxes(:, 1:4:end), im_width), 1);
    % y1 >= 1 & <= im_height
    boxes(:, 2:4:end) = max(min(boxes(:, 2:4:end), im_height), 1);
    % x2 >= 1 & <= im_width
    boxes(:, 3:4:end) = max(min(boxes(:, 3:4:end), im_width), 1);
    % y2 >= 1 & <= im_height
    boxes(:, 4:4:end) = max(min(boxes(:, 4:4:end), im_height), 1);
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file);
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  2.^[-1:5]);%0820:2.^[3:5] -->  2.^[-1:5]
end