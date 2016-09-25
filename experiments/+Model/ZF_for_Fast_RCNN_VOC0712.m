function model = ZF_for_Fast_RCNN_VOC0712(model)

model.solver_def_file        = fullfile('..', 'models', 'fast_rcnn_prototxts', 'ZF', 'solver_30k60k.prototxt');
model.test_net_def_file      = fullfile('..', 'models', 'fast_rcnn_prototxts', 'ZF', 'test.prototxt');

model.net_file               = fullfile('..', 'models', 'pre_trained_models', 'ZF', 'ZF.caffemodel');
model.mean_image             = fullfile('..', 'models', 'pre_trained_models', 'ZF', 'mean_image');

end