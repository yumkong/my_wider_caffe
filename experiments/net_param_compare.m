function net_param_compare()

% ========get the following params at breaking point=========
% layer_num = numel(caffe_solver.net.layer_names);
% layer_paramvalue_t1 = containers.Map();
% for k = 1:layer_num
%     layer_name = caffe_solver.net.layer_names{k};
%     if strcmp(layer_name(1:4), 'conv') && length(layer_name) <= 18
%         layer_paramvalue_t1([layer_name '_w']) = caffe_solver.net.params(layer_name,1).get_data();
%         layer_paramvalue_t1([layer_name '_b']) = caffe_solver.net.params(layer_name,2).get_data();
%     end
% end
% % save result
% save('layer_paramvalue_t1.mat', 'layer_paramvalue_t1');

load('layer_paramvalue_t1.mat');
load('layer_paramvalue_t2.mat');

key_names = layer_paramvalue_t1.keys;
for j = 1:layer_paramvalue_t1.length
    abs_diff = abs(layer_paramvalue_t1(key_names{j}) - layer_paramvalue_t2(key_names{j}));
    fprintf('%s param difference: %.5f\n', key_names{j}, sum(abs_diff(:)));
end

% input_t1 = load('net_inputs_t1.mat');
% input_t2 = load('net_inputs_t2.mat');
% 
% num = length(input_t1.net_inputs);
% 
% for j = 1:num
%     abs_diff = abs(input_t1.net_inputs{j} - input_t2.net_inputs{j});
% 	fprintf('input %d difference: %.5f\n', j, sum(abs_diff(:)));
% end

