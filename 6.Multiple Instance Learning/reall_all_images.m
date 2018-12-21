function image_matrix = reall_all_images(directory)
%%
% (b) read all images from a given directory
% apple images
apple_address = [directory,'\apple\*.jpg'];
apple = dir(apple_address);
apple_num = size(apple,1);
apple_img = cell(apple_num,1);
for i = 1:apple_num
    apple_img{i,1} = imread(strcat(apple(i).folder,'\',apple(i).name));
end
% banana images
banana_address = [directory,'\banana\*.jpg'];
banana = dir(banana_address);
banana_num = size(banana,1);
banana_img = cell(banana_num,1);
for i = 1:banana_num
    banana_img{i,1} = imread(strcat(banana(i).folder,'\',banana(i).name));
end
% concatenating apple images and banana images
image_matrix = [apple_img;banana_img];
end