function hit_rate = save_bbox_result(aboxes, imdb, roidb, res_dir)
    % if res_dir is empty, not save resulting image
    save_image = true;
    if isempty(res_dir)
        save_image = false;
    else
        if ~exist(res_dir, 'dir')
           mkdir(res_dir); 
        end
    end

    im_num = length(aboxes);
    hit_rate = zeros(im_num, 1);
    for j = 1:im_num
        fprintf('Processing image %d / %d\n', j, im_num);
        if save_image
            im = imread(imdb.image_at(j));
            imshow(im);
%             image(im); 
%             axis image;
%             axis off;
%             set(gcf, 'Color', 'white');
            scores = aboxes{j}(:,end);
            %only keep bboxes with clf score >= 0.9
            endNum = sum(scores >= 0.9);
            for i = 1:endNum  % can be changed to any positive number to show different #proposals
                bbox = aboxes{j}(i,1:4);
                rect = [bbox(:, 1), bbox(:, 2), bbox(:, 3)-bbox(:,1)+1, bbox(:,4)-bbox(2)+1];
                rectangle('Position', rect, 'LineWidth', 2, 'EdgeColor', [0 1 0]);
            end

            saveName = sprintf('%s\\img_%d_pro_%d',res_dir, j,endNum);
            export_fig(saveName, '-png', '-a1', '-native');
            fprintf('image %d saved.\n', j);
        end
        % roidb.rois(j).boxes is the gt box coords directly acquired from
        % widerface database
        gt_boxes = roidb.rois(j).boxes;
        all_boxes = aboxes{j}(:,1:4);
        gt_num = size(gt_boxes,1);
        cnt = 0;
        for k = 1:gt_num
            overlap_rate = boxoverlap(all_boxes, gt_boxes(k, :));
            %if gt box is overlapped equal or more than 70%, this gt box is
            %counted as "found"
            if any(overlap_rate >= 0.5)
                cnt = cnt + 1;
            end
        end
        hit_rate(j) = cnt / (gt_num + 1e-6);
    end
end