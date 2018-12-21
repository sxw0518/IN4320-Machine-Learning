function images_matrix = read_images(directory)
images_addres = [directory,'\*.jpg'];
images = dir(images_addres);
images_num = size(images);
images_matrix = cell(images_num);
for i = 1:images_num
    images_matrix{i,1} = imread(strcat(images(i).folder,'\',images(i).name));
end
end