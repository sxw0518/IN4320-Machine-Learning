% load('matlab.mat');
datset_bag = prdataset(bags_data,bags_label);
iteration = 10;
error = zeros(iteration,1);
k =3;
for j = 1:iteration
    % split training and test dataset
    [train_bag,test_bag,train_bag_id,test_bag_id] = gendat(dataset_bag,0.8);
    train_instance_id = [];
    test_instance_id = [];
    for i = 1:size(train_bag,1)
        temp = find(getident(bag_dataset,'milbag')==train_bag_id(i));
        train_instance_id = [train_instance_id;temp];
    end
    train_instance = bag_dataset(train_instance_id,:);
    for i = 1:size(test_bag,1)
        temp = find(getident(bag_dataset,'milbag')==test_bag_id(i));
        test_instance_id = [test_instance_id;temp];
    end
    test_instance = bag_dataset(test_instance_id,:);
    % distance matrix of the training data
    distance_matrix = zeros(size(dataset_bag,1),size(dataset_bag,1)-1);
    for m = 1:size(train_instance,1)
        k = 1;
        for n = 1:size(train_instance,1)
            if i~=j
                distance_matrix(i,k) = HausdorffDist(train_instance{i},train_instance{j});
                k = k+1;
            end
        end
    end
    % test
    test_error = 0;
    for m = 1:size(test_instance,1)
        output = predict(train_instance,distance_matrix,unique(getident(train_instance,'milbag')),test_instance,k,5,5);
        true_label = test_instance.labels;
        if output~=true_label(i)
            test_error = test_error +1;
        end
    end
    test_error = test_error/size(test_instance,1);
end