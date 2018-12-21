function feature_vector = bagembed(bags_data,instance_list)
%%
num_bag = size(bags_data,1);
num_instance = size(instance_list,1);
sigma = 25;
feature_vector = zeros(num_bag,1);
for i = 1:num_bag
    best = 0;
    for j = 1:num_instance
        temp = exp((-1/sigma^2)*sum((bags_data{i,:}-instance_list(j,:)).^2));
        if temp>best
            best = temp;
        end
    end
    feature_vector(i) = best;
end
end