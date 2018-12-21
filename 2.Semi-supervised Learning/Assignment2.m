%%
% question 2
% data normalization
% label: 'g' -> 0; 'h' -> 1
% data = xlsread('data.csv');
% feature = data(:,1:10);
% label = data(:,11);
% [row,column] = size(feature);
% standard_deviations = std(feature);
% for i = 1:column
%     standard_deviations(i) = 1/standard_deviations(i);
% end
% norm_fea = feature * repmat(standard_deviations',1,10);
%%
load('workspace.mat');
%%
% supervised learning - error rate
pr_dataset = prdataset(norm_fea,label);
prwaitbar off;
%%
prwaitbar off;
sup_error = cleval(pr_dataset,ldc,[25,35,45,65,105,185,345,665],50);
% plot(sup_error.xvalues,sup_error.error);
% xlabel(sup_error.xlabel);
% ylabel(sup_error.ylabel);
% legend('supervised error rate');
% hold on;
% scatter(sup_error.xvalues,sup_error.error);
% training_set = [pr_dataset(1:13,:);pr_dataset(row-11:row,:)];
% testing_set = pr_dataset(14:row-12,:);
% ldc_mapping = ldc(pr_dataset(1:13,:));
% test_size = [0,10,20,40,80,160,320];
% times = size(test_size,2);
% sup_error = zeros(1,times);
% for i = 1:times
%     test_d = testing_set(randperm(size(testing_set,1),test_size(i)),:);
%     sup_error(i) = test_d * ldc_mapping * testc([],[]);
% end
%%
% semi-super method 1
% self-learning
glabeled_size = size(find(label == 0),1);
hlabeled_size = size(find(label == 1),1);
% labeled_d = [pr_dataset(1:13,:);pr_dataset(row-11:row,:)];
% unlabeled_d = pr_dataset(14:row-12,:);
unlab_size = [0,10,20,40,80,160,320,640];
times = size(unlab_size,2);
%%
error_1 = zeros(1,times);
for i = 1:50
    idx_g = randperm(glabeled_size);
    labeled_g = pr_dataset(idx_g(1:13),:);
    unlabeled_g = pr_dataset(idx_g(14:end),:);
    idx_h = randperm(hlabeled_size);
    idx_h = idx_h + repmat(glabeled_size,1,hlabeled_size);
    labeled_h = pr_dataset(idx_h(1:12),:);
    unlabeled_h = pr_dataset(idx_h(13:end),:);
    labeled_d = [labeled_g;labeled_h];
    unlabeled_d = [unlabeled_g;unlabeled_h];
    for j = 1:times
        ldc_mapping = ldc(labeled_d);
        if j == 1
            [error,c] = testc(unlabeled_d,ldc_mapping);
            error_1(1,j) = error_1(1,j) + error; 
            continue;
        else
            iteration = 1;
            aver_dis = 1;
            idx = randperm(size(unlabeled_d,1));
            training_un = unlabeled_d(idx(1:unlab_size(j)),:);
            test_d = unlabeled_d(idx(unlab_size(j)+1:end),:);
            training_d = [labeled_d;training_un];
            predicted_lab = getlabels(ldc_mapping(training_un));
            predicted_lab = [labeled_d.labels;predicted_lab];
            training_d = prdataset(getdata(training_d),predicted_lab);
            ldc_mapping = ldc(training_d);
            present_lab = predicted_lab;
            while aver_dis > 10^(-5) && iteration <= 1000
                previous_lab = present_lab;
                idx = randperm(size(unlabeled_d,1));
                training_un = unlabeled_d(idx(1:unlab_size(j)),:);
                test_d = unlabeled_d(idx(unlab_size(j)+1:end),:);
                training_d = [labeled_d;training_un];
                predicted_lab = getlabels(ldc_mapping(training_un));
                predicted_lab = [labeled_d.labels;predicted_lab];
                training_d = prdataset(getdata(training_d),predicted_lab);
                iteration = iteration + 1;
                ldc_mapping = ldc(training_d);
                present_lab = predicted_lab;
                aver_dis = norm(present_lab - previous_lab)/size(test_d,1);
            end
        end
        [error,c] = testc(unlabeled_d,ldc_mapping);
        error_1(1,j) = error_1(1,j) + error; 
    end
end
error_1 = error_1/i;
%%
% semi-super method 2
% step 1: whitening the data
[A,B,C,D] = whiten(norm_fea);
whiten_prdataset = prdataset(A,label);
%%
% step 2: calculate all means and covariance matrix
error_2 = zeros(1,times);
for i = 1:50
    idx_g = randperm(glabeled_size);
    labeled_g = whiten_prdataset(idx_g(1:13),:);
    unlabeled_g = whiten_prdataset(idx_g(14:end),:);
    idx_h = randperm(hlabeled_size);
    idx_h = idx_h + repmat(glabeled_size,1,hlabeled_size);
    labeled_h = whiten_prdataset(idx_h(1:12),:);
    unlabeled_h = whiten_prdataset(idx_h(13:end),:);
    labeled_d = [labeled_g;labeled_h];
    unlabeled_d = [unlabeled_g;unlabeled_h];
    for j = 1:times
        m = mean(getdata(labeled_d));
        T = cov(getdata(labeled_d));
        if j == 1
            ldc_mapping = ldc(labeled_d);
            [error,c] = testc(unlabeled_d,ldc_mapping);
%             true_lab = getlabels(unlabeled_d);
%             predicted_lab = getlabels(ldc_mapping(unlabeled_d));
        else
            idx = randperm(size(unlabeled_d,1));
            training_un = unlabeled_d(idx(1:unlab_size(j)),:);
%             true_lab = getlabels(unlabeled_d);
            mu = mean(getdata(training_un));
            Tu = cov(getdata(training_un));
            new_data = (labeled_d - repmat(m,size(labeled_d,1),1))*Tu.^(1/2) * pinv(T).^(1/2) + repmat(mu,size(labeled_d,1),1);
            ldc_mapping = ldc(new_data);
%             predicted_lab = getlabels(ldc_mapping(unlabeled_d));
            [error,c] = testc(unlabeled_d,ldc_mapping);
        end
        error_2(j) = error_2(j) + error;
    end
end
error_2 = error_2/i;
%%
% plot the learning curves
plot(unlab_size,repmat(sup_error.error(1),times,1));
xlabel('the number of unlabeled samples');
ylabel(sup_error.ylabel);
hold on;
plot(unlab_size,error_1);
plot(unlab_size,error_2);
legend('supervised error rate','semi-supervsied method 1 - self learning error rate','semi-supervsied method 2 - Fishers LDC error rate');
scatter(unlab_size,repmat(sup_error.error(1),times,1));
scatter(unlab_size,error_1);
scatter(unlab_size,error_2);
%% c
% log-likelihood
% method 1
log_1 = zeros(1,times);
for i = 1:50
    idx_g = randperm(glabeled_size);
    labeled_g = pr_dataset(idx_g(1:13),:);
    unlabeled_g = pr_dataset(idx_g(14:end),:);
    idx_h = randperm(hlabeled_size);
    idx_h = idx_h + repmat(glabeled_size,1,hlabeled_size);
    labeled_h = pr_dataset(idx_h(1:12),:);
    unlabeled_h = pr_dataset(idx_h(13:end),:);
    labeled_d = [labeled_g;labeled_h];
    unlabeled_d = [unlabeled_g;unlabeled_h];
    for j = 1:times
        ldc_mapping = ldc(labeled_d);
        if j == 1
            ldc_mapping = ldc(labeled_d);
            trained_data = ldc_mapping(labeled_d);
            labeled_g = labeled_d(trained_data.labels == 0,:);
            labeled_h = labeled_d(trained_data.labels == 1,:);
            log_labeled_g = -size(labeled_g,1)/2*log(2*pi)-size(labeled_g,1)/2*log(1)-1/(2*1)*sum(labeled_g-repmat(mean(labeled_g),size(labeled_g,1),1).^2);
            log_labeled_h = -size(labeled_h,1)/2*log(2*pi)-size(labeled_h,1)/2*log(1)-1/(2*1)*sum(labeled_h-repmat(mean(labeled_h),size(labeled_h,1),1).^2);
            log_unlabeled = -size(unlabeled_d,1)/2*log(2*pi)-size(unlabeled_d,1)/2*log(1)-1/(2*1)*sum(unlabeled_d.data-repmat(mean(unlabeled_d.data),size(unlabeled_d.data,1),1).^2);
            log_1(j) = sum(log_labeled_g) + sum(log_labeled_h) + sum(log_unlabeled);
        else
            iteration = 1;
            aver_dis = 1;
            idx = randperm(size(unlabeled_d,1));
            training_un = unlabeled_d(idx(1:unlab_size(j)),:);
            test_d = unlabeled_d(idx(unlab_size(j)+1:end),:);
            training_d = [labeled_d;training_un];
            predicted_lab = getlabels(ldc_mapping(training_un));
            predicted_lab = [labeled_d.labels;predicted_lab];
            training_d = prdataset(getdata(training_d),predicted_lab);
            ldc_mapping = ldc(training_d);
            present_lab = predicted_lab;
            while aver_dis > 10^(-5) && iteration <= 1000
                previous_lab = present_lab;
                idx = randperm(size(unlabeled_d,1));
                training_un = unlabeled_d(idx(1:unlab_size(j)),:);
                test_d = unlabeled_d(idx(unlab_size(j)+1:end),:);
                training_d = [labeled_d;training_un];
                predicted_lab = getlabels(ldc_mapping(training_un));
                predicted_lab = [labeled_d.labels;predicted_lab];
                training_d = prdataset(training_d.data,predicted_lab);
                iteration = iteration + 1;
                ldc_mapping = ldc(training_d);
                present_lab = predicted_lab;
                aver_dis = norm(present_lab - previous_lab)/size(test_d,1);
            end
            trained_data = ldc_mapping(training_d);
            labeled_g = training_d(trained_data.labels == 0,:);
            labeled_h = training_d(trained_data.labels == 1,:);
            log_labeled_g = -size(labeled_g,1)/2*log(2*pi)-size(labeled_g,1)/2*log(1)-1/(2*1)*sum(labeled_g-repmat(mean(labeled_g),size(labeled_g,1),1).^2);
            log_labeled_h = -size(labeled_h,1)/2*log(2*pi)-size(labeled_h,1)/2*log(1)-1/(2*1)*sum(labeled_h-repmat(mean(labeled_h),size(labeled_h,1),1).^2);
            log_unlabeled = -size(test_d,1)/2*log(2*pi)-size(test_d,1)/2*log(1)-1/(2*1)*sum(test_d.data-repmat(mean(test_d.data),size(test_d.data,1),1).^2);
            log_1(j) = sum(log_labeled_g) + sum(log_labeled_h) + sum(log_unlabeled);
        end
    end
end
log_1 = log_1/i;
%%
% method 2
log_2 = zeros(1,times);
for i = 1:50
    idx_g = randperm(glabeled_size);
    labeled_g = whiten_prdataset(idx_g(1:13),:);
    unlabeled_g = whiten_prdataset(idx_g(14:end),:);
    idx_h = randperm(hlabeled_size);
    idx_h = idx_h + repmat(glabeled_size,1,hlabeled_size);
    labeled_h = whiten_prdataset(idx_h(1:12),:);
    unlabeled_h = whiten_prdataset(idx_h(13:end),:);
    labeled_d = [labeled_g;labeled_h];
    unlabeled_d = [unlabeled_g;unlabeled_h];
    for j = 1:times
        m = mean(getdata(labeled_d));
        T = cov(getdata(labeled_d));
        if j == 1
            ldc_mapping = ldc(labeled_d);
            trained_data = ldc_mapping(labeled_d);
            labeled_g = labeled_d(trained_data.labels == 0,:);
            labeled_h = labeled_d(trained_data.labels == 1,:);
            log_labeled_g = -size(labeled_g,1)/2*log(2*pi)-size(labeled_g,1)/2*log(1)-1/(2*1)*sum(labeled_g-repmat(mean(labeled_g),size(labeled_g,1),1).^2);
            log_labeled_h = -size(labeled_h,1)/2*log(2*pi)-size(labeled_h,1)/2*log(1)-1/(2*1)*sum(labeled_h-repmat(mean(labeled_h),size(labeled_h,1),1).^2);
            log_unlabeled = -size(unlabeled_d,1)/2*log(2*pi)-size(unlabeled_d,1)/2*log(1)-1/(2*1)*sum(unlabeled_d.data-repmat(mean(unlabeled_d.data),size(unlabeled_d.data,1),1).^2);
            log_2(j) = sum(log_labeled_g) + sum(log_labeled_h) + sum(log_unlabeled);
        else
            idx = randperm(size(unlabeled_d,1));
            training_un = unlabeled_d(idx(1:unlab_size(j)),:);
            test_d = unlabeled_d(idx(unlab_size(j)+1:end),:);
            mu = mean(getdata(training_un));
            Tu = cov(getdata(training_un));
            new_data = (labeled_d - repmat(m,size(labeled_d,1),1))*Tu.^(1/2) * pinv(T).^(1/2) + repmat(mu,size(labeled_d,1),1);
            ldc_mapping = ldc(new_data);
            trained_data = ldc_mapping(new_data);
            labeled_g = new_data(trained_data.labels == 0,:);
            labeled_h = new_data(trained_data.labels == 1,:);
            log_labeled_g = -size(labeled_g,1)/2*log(2*pi)-size(labeled_g,1)/2*log(1)-1/(2*1)*sum(labeled_g-repmat(mean(labeled_g),size(labeled_g,1),1).^2);
            log_labeled_h = -size(labeled_h,1)/2*log(2*pi)-size(labeled_h,1)/2*log(1)-1/(2*1)*sum(labeled_h-repmat(mean(labeled_h),size(labeled_h,1),1).^2);
            log_unlabeled = -size(test_d,1)/2*log(2*pi)-size(test_d,1)/2*log(1)-1/(2*1)*sum(test_d.data-repmat(mean(test_d.data),size(test_d.data,1),1).^2);
            log_2(j) = sum(log_labeled_g) + sum(log_labeled_h) + sum(log_unlabeled);
        end
    end
end
log_2 = log_2/i;
%%
% plot the log-likellihood curves
xlabel('the number of unlabeled samples');
ylabel('log-likelihood');
plot(unlab_size,log_1);
hold on
scatter(unlab_size,log_1);
legend('log-likelihood for method 1','Location','southeast');
hold off;
figure;
xlabel('the number of unlabeled samples');
ylabel('log-likelihood');
plot(unlab_size,log_2);
hold on;
legend('log-likelihood for method 2','Location','southeast');
scatter(unlab_size,log_2);
hold off;
%%
% generate the dataset
prwaitbar off;
dataset_1 = gauss([1000 100],[0,0;3,3]*2,cat(3,[2 1; 1 3],eye(2)));
labeled_1 = dataset_1(1:1000,:);
labeled_2 = dataset_1(1001:end,:);
glabeled_size = size(labeled_1,1);
hlabeled_size = size(labeled_2,1);
idx_g = randperm(glabeled_size);
labeled_g = dataset_1(idx_g(1:13),:);
unlabeled_g = dataset_1(idx_g(14:end),:);
idx_h = randperm(hlabeled_size);
idx_h = idx_h + repmat(glabeled_size,1,hlabeled_size);
labeled_h = dataset_1(idx_h(1:12),:);
unlabeled_h = dataset_1(idx_h(13:end),:);
labeled_d = [labeled_g;labeled_h];
unlabeled_d = [unlabeled_g;unlabeled_h];
supervised_ldc = ldc(labeled_d);
[error1_sup,c] = testc(unlabeled_d,supervised_ldc);
%% method 1
ldc_mapping = ldc(labeled_d);
iteration = 1;
aver_dis = 1;
idx = randperm(size(unlabeled_d,1));
training_un = unlabeled_d(idx(1:25),:);
test_d = unlabeled_d(idx(26:end),:);
training_d = [labeled_d;training_un];
predicted_lab = getlabels(ldc_mapping(training_un));
predicted_lab = [labeled_d.labels;predicted_lab];
training_d = prdataset(getdata(training_d),predicted_lab);
ldc_mapping = ldc(training_d);
present_lab = predicted_lab;
while aver_dis > 10^(-5) && iteration <= 1000
    previous_lab = present_lab;
    idx = randperm(size(unlabeled_d,1));
    training_un = unlabeled_d(idx(1:25),:);
    test_d = unlabeled_d(idx(26:end),:);
    training_d = [labeled_d;training_un];
    predicted_lab = getlabels(ldc_mapping(training_un));
    predicted_lab = [labeled_d.labels;predicted_lab];
    training_d = prdataset(getdata(training_d),predicted_lab);
    iteration = iteration + 1;
    ldc_mapping = ldc(training_d);
    present_lab = predicted_lab;
    aver_dis = norm(present_lab - previous_lab)/size(test_d,1);
end
[error1_1,c] = testc(unlabeled_d,ldc_mapping);
% plotc(ldc_mapping,5);
%%
[A,B,C,D] = whiten(dataset_1.data);
whiten_prdataset = prdataset(A,dataset_1.labels);
idx_g = randperm(glabeled_size);
labeled_g = whiten_prdataset(idx_g(1:13),:);
unlabeled_g = whiten_prdataset(idx_g(14:end),:);
idx_h = randperm(hlabeled_size);
idx_h = idx_h + repmat(glabeled_size,1,hlabeled_size);
labeled_h = whiten_prdataset(idx_h(1:12),:);
unlabeled_h = whiten_prdataset(idx_h(13:end),:);
labeled_d = [labeled_g;labeled_h];
unlabeled_d = [unlabeled_g;unlabeled_h];
m = mean(getdata(labeled_d));
T = cov(getdata(labeled_d));
idx = randperm(size(unlabeled_d,1));
training_un = unlabeled_d(idx(1:25),:);
mu = mean(getdata(training_un));
Tu = cov(getdata(training_un));
new_data = (labeled_d - repmat(m,size(labeled_d,1),1))*Tu.^(1/2) * pinv(T).^(1/2) + repmat(mu,size(labeled_d,1),1);
ldc_mapping_2 = ldc(new_data);
[error1_2,c] = testc(unlabeled_d,ldc_mapping_2);
%%
figure;
scatter(labeled_1(:,1),labeled_1(:,2),1);
hold on;
scatter(labeled_2(:,1),labeled_2(:,2),1);
% scatterd(dataset_1);
% hold on;
plotc({supervised_ldc,ldc_mapping,ldc_mapping_2});
% legend('supvised','method 1','method 2');
hold off;
%%
% second dataset
prwaitbar off;
dataset_2 = gauss([1000 1000],[0,0;0,4],cat(3,eye(2),eye(2)));
labeled_1 = dataset_2(find(dataset_2.data(:,1)>0),:);
labeled_2 = dataset_2(find(dataset_2.data(:,1)<=0),:);
label_1 = zeros(size(labeled_1,1),1);
label_2 = ones(size(labeled_2,1),1);
labeled_1 = prdataset(labeled_1,label_1);
labeled_2 = prdataset(labeled_2,label_2);
dataset_2 = [labeled_1;labeled_2];
% labeled_1 = dataset_2(1:1000,:);
% labeled_2 = dataset_2(1001:end,:);
%%
glabeled_size = size(labeled_1,1);
hlabeled_size = size(labeled_2,1);
idx_g = randperm(glabeled_size);
labeled_g = dataset_2(idx_g(1:13),:);
unlabeled_g = dataset_2(idx_g(14:end),:);
idx_h = randperm(hlabeled_size);
idx_h = idx_h + repmat(glabeled_size,1,hlabeled_size);
labeled_h = dataset_2(idx_h(1:12),:);
unlabeled_h = dataset_2(idx_h(13:end),:);
labeled_d = [labeled_g;labeled_h];
unlabeled_d = [unlabeled_g;unlabeled_h];
supervised_ldc = ldc(labeled_d);
[error1_sup,c] = testc(unlabeled_d,supervised_ldc);
%%
%method 1
ldc_mapping = ldc(labeled_d);
iteration = 1;
aver_dis = 1;
idx = randperm(size(unlabeled_d,1));
training_un = unlabeled_d(idx(1:25),:);
test_d = unlabeled_d(idx(26:end),:);
training_d = [labeled_d;training_un];
predicted_lab = getlabels(ldc_mapping(training_un));
predicted_lab = [labeled_d.labels;predicted_lab];
training_d = prdataset(getdata(training_d),predicted_lab);
ldc_mapping = ldc(training_d);
present_lab = predicted_lab;
while aver_dis > 10^(-5) && iteration <= 1000
    previous_lab = present_lab;
    idx = randperm(size(unlabeled_d,1));
    training_un = unlabeled_d(idx(1:25),:);
    test_d = unlabeled_d(idx(26:end),:);
    training_d = [labeled_d;training_un];
    predicted_lab = getlabels(ldc_mapping(training_un));
    predicted_lab = [labeled_d.labels;predicted_lab];
    training_d = prdataset(getdata(training_d),predicted_lab);
    iteration = iteration + 1;
    ldc_mapping = ldc(training_d);
    present_lab = predicted_lab;
    aver_dis = norm(present_lab - previous_lab)/size(test_d,1);
end
[error1_1,c] = testc(unlabeled_d,ldc_mapping);
%%
% method 2
[A,B,C,D] = whiten(dataset_2.data);
whiten_prdataset = prdataset(A,dataset_2.labels);
idx_g = randperm(glabeled_size);
labeled_g = whiten_prdataset(idx_g(1:13),:);
unlabeled_g = whiten_prdataset(idx_g(14:end),:);
idx_h = randperm(hlabeled_size);
idx_h = idx_h + repmat(glabeled_size,1,hlabeled_size);
labeled_h = whiten_prdataset(idx_h(1:12),:);
unlabeled_h = whiten_prdataset(idx_h(13:end),:);
labeled_d = [labeled_g;labeled_h];
unlabeled_d = [unlabeled_g;unlabeled_h];
m = mean(getdata(labeled_d));
T = cov(getdata(labeled_d));
idx = randperm(size(unlabeled_d,1));
training_un = unlabeled_d(idx(1:25),:);
mu = mean(getdata(training_un));
Tu = cov(getdata(training_un));
new_data = (labeled_d - repmat(m,size(labeled_d,1),1))*Tu.^(1/2) * pinv(T).^(1/2) + repmat(mu,size(labeled_d,1),1);
ldc_mapping_2 = ldc(new_data);
[error1_2,c] = testc(unlabeled_d,ldc_mapping_2);
%%
figure;
scatter(labeled_1(:,1),labeled_1(:,2),1);
hold on;
scatter(labeled_2(:,1),labeled_2(:,2),1);
plotc({supervised_ldc,ldc_mapping_2,ldc_mapping});