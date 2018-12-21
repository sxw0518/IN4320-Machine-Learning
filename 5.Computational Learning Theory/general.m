%%
% c
data_a = gauss(100,[0 0]);
data_b = gauss(100,[2 0]);
%%
scatterd(data_a,'r*');
hold on;
scatterd(data_b,'b.');
legend('class 1','class 2')
%%
data_set = [data_a;data_b];
label = ones(size(data_set,1),1);
label(size(data_a,1)+1:end) = 2;
%%
data = prdataset(data_set,label);
%%
[f,theta,y] = weaklearner(data);
%% d
data_d = importdata('optdigitsubset.txt');
%%
data_0 = data_d(1:554,:);
data_1 = data_d(555:end,:);
label_0 = ones(size(data_0,1),1);
label_1 = ones(size(data_1,1),1)+1;
%% _~_1:first 50 for training _~_2:random 50 for training
%%
train_0_1 = data_0(1:50,:);
test_0_1 = data_0(51:end,:);
train_1_1 = data_1(1:50,:);
test_1_1 = data_1(51:end,:);
train_1 = [train_0_1;train_1_1];
label_train_1 = [label_0(1:50);label_1(1:50)];
test_1 = [test_0_1;test_1_1];
label_test_1 = [label_0(51:end);label_1(51:end)];
traindata_1 = prdataset(train_1,label_train_1);
testdata_1 = prdataset(test_1,label_test_1);
%%
[f_1,theta_1,y_1] = weaklearner(traindata_1);
%%
temp = test_1(:,f_1)>=theta_1;
temp = temp+1;
%%
error_1 = sum(abs(temp-label_test_1))/size(test_1,1);
%% randomly selecting 50 from both classes
error = zeros(1,50);
for i=1:50
    train_label = label_train_1;
    index = randperm(size(data_0,1));
    train_0_2 = data_0(index(1:50),:);
    test_0_2 = data_0(index(51:end),:);
    index = randperm(size(data_1,1));
    train_1_2 = data_1(index(1:50),:);
    test_1_2 = data_1(index(51:end),:);
    train_2 = [train_0_2;train_1_2];
    test_2 = [test_0_2;test_1_2];
    label_train_2 = [label_0(1:50);label_1(1:50)];
    label_test_2 = [label_0(51:end);label_1(51:end)];
    traindata_2 = prdataset(train_2,label_train_2);
    [f_2,theta_2,y_2] = weaklearner(traindata_2);
    if y_2 == 0
        temp = test_2(:,f_2)>=theta_2;
        temp = temp+1;
        error(i) = sum(abs(temp-label_test_2))/size(test_2,1);
    else
        temp = test_2(:,f_2)<=theta_2;
        temp = temp+1;
        error(i) = sum(abs(temp-label_test_2))/size(test_2,1);
    end
    
end
%%
m = mean(error);
s = std(error);
%% d
index = randperm(size(data_a,1));
simple_a = data_a(index(1:7),:);
simple_b = data_b(index(6:12),:);
scatterd(simple_a,'r*');
hold on;
scatterd(simple_b,'b.');
%%
legend('class 1','class 2');
%%
simple = [simple_a;simple_b];
label = [ones(size(simple_a,1),1);ones(size(simple_b,1),1)+1];
simple_data = prdataset(simple,label);
[f_d,theta_d,y_d] = weighted_weaklearner(simple_data,ones(1,size(simple,1)));
%%
[f_d1,theta_d1,y_d1] = weighted_weaklearner(simple_data,[2,2,1,2,1,1,2,1,1,1,1,1,1,1]);
%% g
[predict_lab,w] = adaboost(simple_data,100);
%% banana shaped dataset
data_banana = gendatb;
%%
scatterd(data_banana);
legend('class 1','class 2')
hold on;
plot(data_banana.data(2,1),data_banana.data(2,2),'kdiamond');
hold on;
plot(data_banana.data(97,1),data_banana.data(97,2),'ksquare');
%%
[predict_lab,beta,para,w] = adaboost(data_banana,100);
%%
figure;
imagesc(w),colorbar;
% plot(w)
%%
X = gendatb(50,2);
lab= str2num(getlab(data_banana));
X_data = getdata(data_banana);
X_train = prdataset(X_data,lab);
figure;
scatterd(X_train,'legend')
T=100;
%train the classifier
[predLab,beta,para,W] = adaboost(X_train,T);
error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1)
figure;
imagesc(W'),colorbar;
%%
%% testing for digit datset
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);

%scatterd(X,'legend')
T=5;
%train the classifier
[predLab,beta,para] = adaBoost(X_train,T);
error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1)
% on test set
predLab_test = adaPredict(beta,para,X_test);
error_test = sum(abs(predLab_test-(getlab(X_test))))/size(X_test,1)
%%
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);
E_test_history = [];
t_list = [1 5 10 20:2:40 100];
t_list = [1 2 3 5 10 12 14 16 18 20 25 30 35 40];
t_list = [1 10 30 40 50]
for t = t_list
    [predLab,beta,para] = adaBoost(X_train,t);
    error_train = sum(abs(predLab-str2num(getlab(X_train))))/size(X_train,1);
    predLab_test = adaPredict(beta,para,X_test);
    error_test = sum(abs(predLab_test-str2num(getlab(X_test))))/size(X_test,1);
    E_test_history = [E_test_history error_train];
end
plot(t_list,E_test_history)

%% test
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);
T = 200;
E_test_history = zeros(T,1);
E_trn_history = zeros(T,1);
[predLab,beta,para] = adaBoost(X_train,T);
for t = 1:T
    predLab = adaPredict(beta(1:t),para(1:t,:),X_train);
    error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1);
    predLab_test = adaPredict(beta(1:t),para(1:t,:),X_test);
    error_test = sum(abs(predLab_test-(getlab(X_test))))/size(X_test,1);
    E_test_history(t) = error_test;
    E_trn_history(t) = error_train;
end
%%
plot(E_test_history(1:100));
hold on
plot(E_trn_history(1:100));
legend('test','train');
xlabel('T:iterations');
ylabel('error');
title('Testing and training error on different T');

%% Optimal T: 17
[predLab,beta,para,W] = adaBoost(X_train,17);
imagesc(W'),colorbar;

%% show the outlier
%% testing for digit datset
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);

%scatterd(X,'legend')
T=5;
%train the classifier
[predLab,beta,para] = adaBoost(X_train,T);
error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1)
% on test set
predLab_test = adaPredict(beta,para,X_test);
error_test = sum(abs(predLab_test-(getlab(X_test))))/size(X_test,1)
%%
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);
E_test_history = [];
t_list = [1 5 10 20:2:40 100];
t_list = [1 2 3 5 10 12 14 16 18 20 25 30 35 40];
t_list = [1 10 30 40 50]
for t = t_list
    [predLab,beta,para] = adaBoost(X_train,t);
    error_train = sum(abs(predLab-str2num(getlab(X_train))))/size(X_train,1);
    predLab_test = adaPredict(beta,para,X_test);
    error_test = sum(abs(predLab_test-str2num(getlab(X_test))))/size(X_test,1);
    E_test_history = [E_test_history error_train];
end
plot(t_list,E_test_history)

%% test
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);
T = 200;
E_test_history = zeros(T,1);
E_trn_history = zeros(T,1);
[predLab,beta,para] = adaBoost(X_train,T);
for t = 1:T
    predLab = adaPredict(beta(1:t),para(1:t,:),X_train);
    error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1);
    predLab_test = adaPredict(beta(1:t),para(1:t,:),X_test);
    error_test = sum(abs(predLab_test-(getlab(X_test))))/size(X_test,1);
    E_test_history(t) = error_test;
    E_trn_history(t) = error_train;
end
%%
plot(E_test_history(1:100));
hold on
plot(E_trn_history(1:100));
legend('test','train');
xlabel('T:iterations');
ylabel('error');
title('Testing and training error on different T');

%% Optimal T: 17
[predLab,beta,para,W] = adaboost(X_train,17);
plot(W)

