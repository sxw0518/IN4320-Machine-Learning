%% not trustworthy
instance = bag_dataset.data;
fisher = fisherc(bag_dataset);
start = 1;
predict = [];
for i = 1:size(bags_data,1)
    instance_length = size(bags_data{i},1);
    instance_bag = instance(start:start+instance_length-1,:);
    start = start+instance_length;
    result = labeld(instance_bag,fisher);
    out_label = combineinstlabels(result);
    predict = [predict; out_label];
end
residule = bags_label - predict;
% apple misclassified as banana
apple_error = sum(residule == -1);
% banana misclassified as apple
banana_error = sum(residule == 1);
%% trustworthy
% split the dataset with bags ID, and then use instances of related bags ID
% to train and test the classifier
bag_id = [1:size(bags_data,1)]';
dataset_bag = prdataset(bag_id,bags_label);
error = cell(2,10);
for j = 1:size(error,2)
    [train_bag,test_bag,train_bag_id,test_bag_id] = gendat(dataset_bag,0.8);
    train_instance_id = [];
    test_instance_id = [];
    for i = 1:size(train_bag,1)
        temp = find(getident(bag_dataset,'milbag')==train_bag_id(i));
        train_instance_id = [train_instance_id;temp];
    end
    train_instance = bag_dataset(train_instance_id,:);
    fisher = fisherc(train_instance);
    predict = [];
    for i = 1:size(test_bag,1)
        temp = find(getident(bag_dataset,'milbag')==test_bag_id(i));
        result = labeld(bag_dataset.data(temp,:),fisher);
        out_label = combineinstlabels(result);
        predict = [predict; out_label];
    end
    test_label = test_bag.labels;
    residule = test_label - predict;
    %apple error
    error{1,j} = sum(residule == -1);
    %banana error
    error{2,j} = sum(residule == 1);
end




