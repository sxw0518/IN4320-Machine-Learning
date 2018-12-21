function bag_dataset = gendatmilsival(directory)
%%
bags_data = cell(0);
bags_label = [];
width = 30;
% apple images
apple_directory = [directory,'\apple'];
apple_images_matrix = read_images(apple_directory);
for i = 1:size(apple_images_matrix,1)
    instances = extractinstances(apple_images_matrix{i},width);
    bags_data{i,1} = instances;
end
% banana images
banana_directory = [directory,'\banana'];
banana_images_matrix = read_images(banana_directory);
for i = 1:size(banana_images_matrix,1)
    instances = extractinstances(banana_images_matrix{i},width);
    bags_data{i+size(banana_images_matrix,1),1} = instances;
end
% label
bags_label = [ones(size(apple_images_matrix,1),1);2*ones(size(banana_images_matrix,1),1)];
% storing them in a Prtools dataset
bag_dataset = bags2dataset(bags_data,bags_label);
end