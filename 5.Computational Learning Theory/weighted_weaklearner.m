function [f,theta,y] = weighted_weaklearner(pr_dataset,weight)
% This function 'weaklearner' is for IN4320 Computational Learning Theory
% question e: implement a 'weighted weak learner'
% This decision stump is for finding the optimal f, theta and sign y for
% which the classification error on the training set is minimum.
% The input of this function is prdataset, which contains data and label,
% weights for updating the error score.
% The output of this function are the optimal f, theta and sign y.
data = getdata(pr_dataset);
label = getlab(pr_dataset);
[num, feature] = size(data);
%num represents the number of data sample, feature represents the number of features
threshold = Inf;
for i = 1:feature
    for j = 1:num
        theta_temp = data(j,i); 
        % theta is selected as the value of each feature of each sample,
        sign1 = data(:,i)>=theta_temp; % if data<theta_temp, 0; otherwise, 1: '<'
        sign2 = data(:,i)<=theta_temp; % if data>theta_temp, 0; otherwise, 1: '>'
        % since the labels are 1 and 2
        score1 = weight*abs(sign1+1-label);
        score2 = weight*abs(sign2+1-label);
        % To select '<' or '>'
        if score1<score2
            score = score1;
            y_temp = 0; % '<'
        else
            score = score2;
            y_temp =1; % '>'
        end
        % Comparing the current error with the smallest error from before
        if score<threshold
            threshold = score;
            y = y_temp;
            f = i;
            theta = theta_temp; 
            % the optimal theta here is not the actual optimal one, but is
            % closed to it
        end
    end
end
end