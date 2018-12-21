function [predicted,w] = ada_boost(pr_dataset,iteration)
% This function 'adaboost' is for IN4320 Computational Learning Theory
% question f: implement 'AsaBoost'
% The input of this function is prdataset, which contains data and label,
% and the time of iteration.
% The output of this function are the predicted label and the weights of
% different objects.
data = getdata(pr_dataset);
label = getlab(pr_dataset);
[num, feature] = size(data);
w = ones(1,num);
beta = zeros(iteration,1);
parameter = zeros(iteration,3);
% iterating T times and calculating T weighted weaklearner
for i = 1:iteration
    p = w./sum(w);
    [f,theta,y] = weighted_weaklearner(pr_dataset,p);
    parameter(i,:) = [f,theta,y];
    if y==0
        predict = data(:,f)>=theta;% if data<theta, 0; otherwise, 1: '<'
        error = w*abs(predict+1-label);
    else
        predict = data(:,f)<=theta;% if data>theta, 0; otherwise, 1: '>'
        error = w*abs(predict+1-label);
    end
    if error == 0 % in case weight is zero all the time
        error = 0.001;
    end
    beta(i) = error/(1-error);
    w = w.*(beta(i).^(1-abs(predict-label)))';
end
temp = 1/(2*sum(log(1./beta))); % from the paper hypothesis
scores = zeros(num,iteration);
for i = 1:iteration
    f = parameter(i,1);
    theta = parameter(i,2);
    y = parameter(i,3);
    if y == 0
        scores(:,i) = data(:,f)>=theta;
    else
        scores(:,i) = data(:,f)<=theta;
    end
    scores(:,i) = scores(:,i).*log(1./beta(i));
end
predicted = sum(scores,2)>=temp; % making labels as 1 and 2
predicted = predicted+1;
end