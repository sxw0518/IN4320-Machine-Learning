% Exercise: Aggregating Algorithm (AA)

clear all;
load coin_data;

d_expert = 5;
n = 213;

% compute adversary movez z_t
z = -log(r);

% compute strategy p_t (see slides)
p = zeros(size(s));
p(1,:) = [0.2, 0.2, 0.2, 0.2, 0.2];
p(2,:) = exp(-z(1,:))./sum(exp(-z(1,:)));
for i = 3:n
    numerator = exp(-sum(z(1:i-1,:)));
    denominator = sum(exp(-sum(z(1:i-1,:))));
    p(i,:) = numerator./denominator;
end

% compute loss of strategy p_t
tmp = p.*exp(-z);
tmp = sum(tmp,2);
for i = 1:n
    loss(i,:) = -log(tmp(i,:));
end
loss_p = sum(loss);

% compute losses of experts
loss = sum(z);
loss_e = min(loss);

% compute regret
R = loss_p-loss_e;

% compute total gain of investing with strategy p_t
gain = sum(p(n,:).*s(n,:)-p(1,:).*s(1,:));

%% plot of the strategy p and the coin data

% if you store the strategy in the matrix p (size n * d)
% this piece of code will visualize your strategy

figure
subplot(1,2,1);
plot(p)
legend(symbols_str)
title('rebalancing strategy AA')
xlabel('date')
ylabel('confidence p_t in the experts')

subplot(1,2,2);
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')
