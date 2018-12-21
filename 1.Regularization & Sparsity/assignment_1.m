%%
% Some Optima & Some Geometry
% 1a
r = [-1:.001:2];
L0 = Loss_1(0,r);
L1 = Loss_1(1,r);
L2 = Loss_1(2,r);
L3 = Loss_1(3,r);

figure
plot(r,L0,r,L1,r,L2,r,L3,r,zeros(1,numel(r)))
legend('\lambda = 0','\lambda = 1','\lambda = 2','\lambda = 3');
grid on;
ylabel('Loss')
xlabel('r_+')
%%
% 1b
x0 = ldm(0,r);
x1 = ldm(1,r);
x2 = ldm(2,r);
x3 = ldm(3,r);
    
figure
plot(r,x0,r,x1,r,x2,r,x3,r,zeros(1,numel(r)))    
legend('\lambda = 0','\lambda = 1','\lambda = 2','\lambda = 3');
grid on;
ylabel('dL/dm')
xlabel('r_+')
%%
% 3a
rp = [-10:.01:10];
rm = [-10:.01:10];
[rp,rm] = meshgrid(rp,rm);
x0 = Lossfunction(rp,rm);
x1 = line1(2,rp,rm);
% x2 = line1(1000,rp,rm);
n = 10;
[c,h] = contour(rp,rm,x0+x1,n);
hold on
[c,h] = contour(rp,rm,x1,n);
% figure;
% [c,h] = contour(rp,rm,x2,n);
hold on
plot([-10,10],[-10,10]);
xlabel('r_+')
ylabel('r_-')
%%
%4b
% lambda = 0
ex=importdata('optdigitsubset.txt');
rp = zeros(1,64);
rm = zeros(1,64);
rp_old = ones(1,64);
rm_old = ones(1,64);
a0 = ex(1:554,1:64);
a1 = ex(555:1125,1:64);
b0 = sum(a0);
b1 = sum(a1);
while abs(lossfunction3(rp,rm,a0,a1,0)-lossfunction3(rp_old,rm_old,a0,a1,0))>2
    rp_old = rp;
    rp = rp -0.0001.*dlmp(rp,b0,rm,0);
    rm_old = rm;
    rm = rm -0.0001.*dlmm(rm,b1,rp_old,0);
end
loss = lossfunction3(rp,rm,a0,a1,0);
mplus = reshape(rp,[8,8]);
mminus = reshape(rm,[8,8]);
mpluss = imresize(mplus,[8,8])/255;
mminuss = imresize(mminus,[8,8])/255;
imshow(mpluss','InitialMagnification','fit');
figure;
imshow(mminuss','InitialMagnification','fit');
%%
% lambda is large enough
figure;
ex=importdata('optdigitsubset.txt');
mp = zeros(1,64);
mm = zeros(1,64);
mp_old = ones(1,64);
mm_old = ones(1,64);
a0 = ex(1:554,1:64);
a1 = ex(555:1125,1:64);
b0 = sum(a0);
b1 = sum(a1);
while abs(lossfunction3(mp,mm,a0,a1,100000)-lossfunction3(mp_old,mm_old,a0,a1,100000))>2000
    mp_old = mp;
    mp = mp -0.0001.*dlmp(mp,b0,mm,100000);
    mm_old = mm;
    mm = mm -0.0001.*dlmm(mm,b1,mp_old,100000);
end
loss = lossfunction3(mp,mm,a0,a1,0);
mplus = reshape(mp,[8,8]);
mminus = reshape(mm,[8,8]);
mpluss = imresize(mplus,[8,8])/255;
mminuss = imresize(mminus,[8,8])/255;
imshow(mpluss','InitialMagnification','fit')
figure;
imshow(mminuss','InitialMagnification','fit')
%%
%4c
% splitting the training examples and test examples
% for 50 times, each time, different training examples are chosen
ex=importdata('optdigitsubset.txt');
true_error = zeros(1,6);
error = zeros(1,6);
for i = 1:50
    a0_train = ex(i,1:64);
    a1_train = ex(554+i,1:64);
    a0_test = [ex(1:i-1,1:64);ex(i+1:554,1:64)];
    a1_test = [ex(555:553+i,1:64);ex(555+i:1125,1:64)];
    lambda = [0,0.1,1,10,100,1000];
    for j = 1:6
        rp = zeros(1,64);
        rm = zeros(1,64);
        rp_old = ones(1,64);
        rm_old = ones(1,64);
        flag = 1;
        lambda1 = lambda(j);
%         lossloss = lossfunction_4c(rp,rm,a0_train,a1_train,lambda(j))-lossfunction_4c(rp_old,rm_old,a0_train,a1_train,lambda(j));
        while abs(lossfunction_4c(rp,rm,a0_train,a1_train,lambda(j))-lossfunction_4c(rp_old,rm_old,a0_train,a1_train,lambda(j)))>min(exp(lambda(j))+1,5)
%             loss = lossfunction_4c(rp,rm,a0_train,a1_train,lambda(j))-lossfunction_4c(rp_old,rm_old,a0_train,a1_train,lambda(j));
%             criterion = min(exp(lambda(j))+1,1000);
            rp_old = rp;
            rp = rp -0.0001.*dlrp(rp,a0_train,rm,0);
            rm_old = rm;
            rm = rm -0.0001.*dlrm(rm,a1_train,rp_old,0);
            flag = flag + 1;
        end
%         mplus = reshape(rp,[8,8]);
%         mminus = reshape(rm,[8,8]);
%         mpluss = imresize(mplus,[8,8])/255;
%         mminuss = imresize(mminus,[8,8])/255;
%         figure;
%         imshow(mpluss','InitialMagnification','fit')
%         figure;
%         imshow(mminuss','InitialMagnification','fit')
        count0 = 0;
        for p = 1:553
            distance1 = sum(sqrt(a0_test(p,1:64).^2 + rp.^2));
            distance2 = sum(sqrt(a0_test(p,1:64).^2 + rm.^2));
            if distance1 > distance2
                count0 = count0 + 1;
            end
            p = p + 1;
        end
        count1 = 0;
        for m = 1:570
            distance1 = sum(sqrt(a1_test(m,1:64).^2 + rp.^2));
            distance2 = sum(sqrt(a1_test(m,1:64).^2 + rm.^2));
            if distance1 < distance2
                count1 = count1 + 1;
            end
            m = m + 1;
        end
        error(j) = (count0 + count1)/(553 + 570);
    end
    true_error = true_error + error;
end
true_error = true_error/50;
% true_error = mean(true_error');
% for i=1:100
%     i = i + 1;
%     rp_old = rp;
%     rp = rp - 0.1.*dlrp(rp,a0_train,rm,lamda);
%     rm = rm - 0.1.*dlrm(rm,a1_train,rp_old,lamda);
% end
% mplus = reshape(rp,[8,8]);
% mminus = reshape(rm,[8,8]);
% mpluss = imresize(mplus,[8,8])/255;
% mminuss = imresize(mminus,[8,8])/255;
% imshow(mpluss','InitialMagnification','fit')
% figure;
% imshow(mminuss','InitialMagnification','fit')
% %%
% % test on image 0
% for i = 1:553
%     
% end
%%
% plot true_error
plot(lambda,true_error,'o-');
hold on
plot(lambda,[0,0,0,0,0,0],'--');
% set(gca,'yTick',[0:0.15:0.3]);
% xt = get(gca,'XTick');
% set(gca,'XTick',xt,'XTickLabel',[0,10.^xt(2:-1)])
legend('True error','Apparent error','Location','best');
grid on;
xlabel('\lambda');
ylabel('Error rate');
