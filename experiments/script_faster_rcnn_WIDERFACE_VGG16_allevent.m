function script_faster_rcnn_WIDERFACE_VGG16_allevent()
% script_faster_rcnn_VOC2007_VGG16()
% Faster rcnn training and testing with VGG16 model
% --------------------------------------------------------
% Faster R-CNN
% Copyright (c) 2015, Shaoqing Ren
% Licensed under The MIT License [see LICENSE for details]
% --------------------------------------------------------

clc;
clear mex;
clear is_valid_handle; % to clear init_key
run(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'startup'));
%% -------------------- CONFIG --------------------
% 0925: added multi-choices for different os
if ispc
    opts.caffe_version          = 'caffe_faster_rcnn';
elseif isunix
    opts.caffe_version          = 'caffe_faster_rcnn_linux';
end
opts.gpu_id                 = auto_select_gpu;
active_caffe_mex(opts.gpu_id, opts.caffe_version);

% do validation, or not 
opts.do_val                 = true; 
% model
model                       = Model.VGG16_for_Faster_RCNN_WIDERFACE;
% cache base
cache_base_proposal         = 'faster_rcnn_WIDERFACE_vgg_16layers_allevent';
cache_base_fast_rcnn        = '';
% train/test data
dataset                     = [];
use_flipped                 = false;  %true --> false
event_num = -1; %3
dataset                     = Dataset.widerface_all(dataset, 'train', use_flipped, event_num);
dataset                     = Dataset.widerface_all(dataset, 'test', false, event_num);

%0805 added, make sure imdb_train and roidb_train are of cell type
if ~iscell(dataset.imdb_train)
    dataset.imdb_train = {dataset.imdb_train};
end
if ~iscell(dataset.roidb_train)
    dataset.roidb_train = {dataset.roidb_train};
end

%% -------------------- TRAIN --------------------
% conf
%conf_proposal               = proposal_config('image_means', model.mean_image, 'feat_stride', model.feat_stride);
conf_proposal               = proposal_config_widerface('image_means', model.mean_image, 'feat_stride', model.feat_stride);
%conf_fast_rcnn              = fast_rcnn_config('image_means', model.mean_image);
% set cache folder for each stage
model                       = Faster_RCNN_Train.set_cache_folder(cache_base_proposal, cache_base_fast_rcnn, model);
% generate anchors and pre-calculate output size of rpn network 
[conf_proposal.anchors, conf_proposal.output_width_map, conf_proposal.output_height_map] ...
                            = proposal_prepare_anchors(conf_proposal, model.stage1_rpn.cache_name, model.stage1_rpn.test_net_def_file);

%%  stage one proposal
fprintf('\n***************\nstage one proposal \n***************\n');
% train
model.stage1_rpn            = Faster_RCNN_Train.do_proposal_train(conf_proposal, dataset, model.stage1_rpn, opts.do_val);
% test
%train_file_suffix = '_train_vgg16_event3';
%train_save_dir = 'train_vgg16_event3_res';
%test_file_suffix = '_val_vgg16_event3';
%test_save_dir = 'val_vgg16_event3_res';
%dataset.roidb_train         = cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, x, y, train_file_suffix, train_save_dir), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
%dataset.roidb_test       	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage1_rpn, dataset.imdb_test, dataset.roidb_test, test_file_suffix, test_save_dir);

% liu@0816 masked --> not necessary currently
% %%  stage one fast rcnn
% fprintf('\n***************\nstage one fast rcnn\n***************\n');
% % train
% model.stage1_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, dataset, model.stage1_fast_rcnn, opts.do_val);
% % test
% opts.mAP                    = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, model.stage1_fast_rcnn, dataset.imdb_test, dataset.roidb_test);
% 
% %%  stage two proposal
% % net proposal
% fprintf('\n***************\nstage two proposal\n***************\n');
% % train
% model.stage2_rpn.init_net_file = model.stage1_fast_rcnn.output_model_file;
% model.stage2_rpn            = Faster_RCNN_Train.do_proposal_train(conf_proposal, dataset, model.stage2_rpn, opts.do_val);
% % test
% dataset.roidb_train        	= cellfun(@(x, y) Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, x, y), dataset.imdb_train, dataset.roidb_train, 'UniformOutput', false);
% dataset.roidb_test         	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);
% 
% %%  stage two fast rcnn
% fprintf('\n***************\nstage two fast rcnn\n***************\n');
% % train
% model.stage2_fast_rcnn.init_net_file = model.stage1_fast_rcnn.output_model_file;
% model.stage2_fast_rcnn      = Faster_RCNN_Train.do_fast_rcnn_train(conf_fast_rcnn, dataset, model.stage2_fast_rcnn, opts.do_val);
% 
% %% final test
% fprintf('\n***************\nfinal test\n***************\n');
%      
% model.stage2_rpn.nms        = model.final_test.nms;
% dataset.roidb_test       	= Faster_RCNN_Train.do_proposal_test(conf_proposal, model.stage2_rpn, dataset.imdb_test, dataset.roidb_test);
% opts.final_mAP              = Faster_RCNN_Train.do_fast_rcnn_test(conf_fast_rcnn, model.stage2_fast_rcnn, dataset.imdb_test, dataset.roidb_test);

% % save final models, for outside tester
% Faster_RCNN_Train.gather_rpn_fast_rcnn_models(conf_proposal, conf_fast_rcnn, model, dataset);
end

function [anchors, output_width_map, output_height_map] = proposal_prepare_anchors(conf, cache_name, test_net_def_file)
    output_map_save_name = fullfile('cache_data', 'output_map_conv5.mat');
    [output_width_map, output_height_map] ...                           
                                = proposal_calc_output_size(conf, test_net_def_file, output_map_save_name);
    anchors                = proposal_generate_anchors(cache_name, ...
                                    'scales',  2.^[-1:5]);%0820:2.^[3:5] -->  2.^[-1:5]
end