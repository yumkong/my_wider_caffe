function model_compare(caffe_net, mat_net_path)

net = load(mat_net_path);
caffe_layer_pool = {'conv1_1','conv1_2', 'conv2_1', 'conv2_2','conv3_1','conv3_2','conv3_3','conv4_1',...
                    'conv4_2', 'conv4_3', 'conv5_1','conv5_2','conv5_3','conv_proposal1',...
                    'proposal_cls_score', 'proposal_bbox_pred'};
format long
for k = 1:16
    layer_name = caffe_layer_pool{k};
    compute_param_diff(caffe_net, net.net_, layer_name, 2*k - 1, 2*k);
end

function compute_param_diff(caffe_net, mat_net, caffe_layer_name, mat_w_idx, mat_b_idx)
    caffe_w = caffe_net.params(caffe_layer_name,1).get_data();
    caffe_b = caffe_net.params(caffe_layer_name,2).get_data();
    matcov_w = mat_net.params(mat_w_idx).value;
%     if mat_w_idx == 1
%         matcov_w = matcov_w(:,:,[3,2,1],:);
%     end
%     matcov_w = permute(matcov_w, [2,1,3,4]);
    matcov_b = mat_net.params(mat_b_idx).value;
    diff_w = abs(matcov_w - caffe_w);
    res_w = sum(diff_w(:));
    diff_b = abs(matcov_b - caffe_b);
    res_b = sum(diff_b(:));
    fprintf('%s: res_w = %.10f, res_b = %.10f\n', caffe_layer_name, res_w, res_b);