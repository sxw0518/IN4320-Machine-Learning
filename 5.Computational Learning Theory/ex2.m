%% testing for digit datset
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);

%scatterd(X,'legend')
T=5;
%train the classifier
[predLab,beta,para] = adaboost(X_train,T);
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
    [predLab,beta,para] = adaboost(X_train,t);
    error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1);
    predLab_test = adaPredict(beta,para,X_test);
    error_test = sum(abs(predLab_test-(getlab(X_test))))/size(X_test,1);
    E_test_history = [E_test_history error_train];
end
plot(t_list,E_test_history)

%% test
X = load('optdigitsubset.txt');
lab = [ones(554,1);ones(571,1)+1];
X = prdataset(X,lab);
X_train = X([1:50,555:604],:);
X_test = X([51:554,605:end],:);
T = 100;
E_test_history = zeros(T,1);
E_trn_history = zeros(T,1);
[predLab,beta,para] = adaboost(X_train,T);
for t = 1:T
    predLab = adaPredict(beta(1:t),para(1:t,:),X_train);
    error_train = sum(abs(predLab-(getlab(X_train))))/size(X_train,1);
    predLab_test = adaPredict(beta(1:t),para(1:t,:),X_test);
    error_test = sum(abs(predLab_test-(getlab(X_test))))/size(X_test,1);
    E_test_history(t) = error_test;
    E_trn_history(t) = error_train;
end
%%
plot(E_test_history(1:100),'r');
% hold on
% plot(E_trn_history(1:100));
xlabel('time of iterations');
ylabel('test error');
%title('Testing and training error on different T');

%% Optimal T: 17
[predLab,beta,para,W] = adaboost(X_train,17);
% imagesc(W'),colorbar;
plot(W)
xlabel('objects number');
ylabel('weight');
%% show the outlier
X_data = getdata(X_train);
% 66 86
index  =88;
img = reshape(X_data(index,:),[8,8]);
img = mat2gray(img');
figure;
show(img)