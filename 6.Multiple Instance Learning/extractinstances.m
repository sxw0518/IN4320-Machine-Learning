function resulting_features = extractinstances(image,width)
%%
% This function is for 1(c) of Multiple Instance Learning: image
% classification of In4320 Machine Learning Computer Exercise.
% Segment an image, compute the average red, green and blue color per
% segment
% use im_meanshift to segment an image
im = im_meanshift(image,width);
% plot the segments to find the width parameter
% imshow(im,[]);
seg_num = size(unique(im),1);
resulting_features = zeros(seg_num,3); % 3 channels: RBG
% compute mean for each segment
segment_cell = cell(0);
segment_idx = [];
segment_num = [];
img = im2double(image);
for i = 1:size(im,1)
    for j = 1:size(im,2)
        if ~ismember(im(i,j),segment_idx)
            segment_idx = [segment_idx im(i,j)];
            segment_num = [segment_num 1];
            segment_cell{size(segment_cell,2)+1} = reshape(img(i,j,:),1,3);
        else
            idx = find(segment_idx == im(i,j));
            segment_cell{idx} = segment_cell{idx} + reshape(img(i,j,:),1,3);
            segment_num(idx) = segment_num(idx) + 1;
        end
    end
end
% return the resulting_features matrix
for i = 1:seg_num
    segment_cell{i} = segment_cell{i}./segment_num(i);
    for j = 1:3 % rgb channels
        resulting_features(i,j) = segment_cell{i}(1,j);
    end
end
end