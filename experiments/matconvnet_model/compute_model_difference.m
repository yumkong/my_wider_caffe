%function compute_model_difference()
%Goal: calulate the parameter difference between modelA and modelB, where modelA is vgg-16 fine-tuned by widerface faster-rcnn and modelB is original vgg-16.

%code:
run ../../matlab/vl_setupnn;
load('res_0902/net-epoch-17.mat');   % modelA

net1 = load('data/models/imagenet-vgg-verydeep-16.mat');
net1 = vl_simplenn_tidy(net1);
net1 = dagnn.DagNN.fromSimpleNN(net1, 'canonicalNames', true);  % modelB
for i = 1:26
    abs_diff = abs(net.params(i).value - net1.params(i).value);
    %res1 = res.^2;
    %sum1 = sum(res1(:));
    fprintf('param name: %s, square diff: %.3f\n', net.params(i).name, sum(abs_diff(:)));
end