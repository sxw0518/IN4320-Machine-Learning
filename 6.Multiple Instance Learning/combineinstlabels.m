function out_label = combineinstlabels(list_label)
labels = unique(list_label);
num = length(labels);
best_label_num = 0;
for i = 1:num
    if best_label_num < sum(list_label==labels(i))
        out_label = labels(i);
        best_label_num = sum(list_label==labels(i));
    end
end
end