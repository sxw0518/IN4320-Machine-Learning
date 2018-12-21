function output = predict(dataset, distance_matrix,bag_id,test_instance,k,R,C)
% This function labels a new bag
bags_num = size(dataset,1);
idx_bags = unique(getident(dataset,'milbag')),1);
list_dist = [];
% test data distances
for i = 1:bags_num
    res = HausdorffDist(test_instance,dataset(find(getident(dataset,'milbag') == idx_bags(i)),:),k);
    list_dist = [list_dist res];
end
% compute references
[~,max_id] = sort(list_dist,'ascend');
id_r = [];
for i = 1:R
    list_r_idx = find(getident(dataset,'milbag')==bag_id(max_id(i)));
    id_r = [id_r list_r_idx];
end
r_lab = dataset(id_r,:).labels;
% compute citers
c_lab = [];
for i=1:idx_bags
    res = HausdorffDist(dataset(find(getident(dataset,'milbag')==idx_bags(i)),:),test_instance,k);
    idx = find(test_instance==bag_id(i));
    list_dist = cat(2,distance_matrix(idx,:),res);
    [max_value,max_id] = sort(list_dist,'ascend');
    rank = find(max_value==res);
    if rank<=c
        id_r = find(getident(dataset,'milbag')==(idx_bags(i)));
        ref = getlab(dataset(id_r(1),:));
        c_lab = [c_lab;ref];
    end
end
% label
nb_1=0;
nb_2=0;
for i = 1:size(c_lab,1)
    if c_lab(i) == 1
        nb_1 = nb_1+1;
    else
        nb_2 = nb_2+1;
    end
end
for i = 1:size(r_lab,1)
    if r_lab(i) == 1
        nb_1 = nb_1 +1;
    else
        nb_2 = nb_2+1;
    end
end
if nb_2>nb_1
    output = 2;
else
    output = 1;
end
end