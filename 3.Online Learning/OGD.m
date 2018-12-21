% Exercise: Online Gradient Descent (OGD)

clear all;
load coin_data;

a_init = [0.2, 0.2, 0.2, 0.2, 0.2]'; % initial action

n = 213; % is the number of days
d = 5; % number of coins

% we provide you with values R and G.
alpha = sqrt(max(sum(r.^2,2))); 
epsilon = min(min(r)); 
G = alpha/epsilon; 
R = 1; 

% set eta:
eta = R/(G*sqrt(n));

a = a_init; % initialize action. a is always a column vector

L = nan(n,1); % keep track of all incurred losses
A = nan(d,n); % keep track of all our previous actions

for t = 1:n
    
    % we play action a
    [l,g] = mix_loss(a,r(t,:)'); % incur loss l, compute gradient g
    
    A(:,t) = a; % store played action
    L(t) = l; % store incurred loss
    
    % update our action, make sure it is a column vector
    a = a-eta.*g;
    
    % after the update, the action might not be anymore in the action
    % set A (for example, we may not have sum(a) = 1 anymore). 
    % therefore we should always project action back to the action set:
    a = project_to_simplex(a')'; % project back (a = \Pi_A(w) from lecture)

end

% compute total loss
loss = sum(L);

% compute total gain in wealth
gain = s(n,:)*A(:,n)-s(1,:)*A(:,1);

% compute best fixed strategy (you may make use of loss_fixed_action.m and optimization toolbox if needed)
loss_fixed = zeros(size(L));
% looking for the best fixed action in matrix A
for i = 1:n
    [loss_fixed(i),gradient] = loss_fixed_action(A(:,i));
end
% looking for the best fixed action out of matrix A
y = zeros(d,15000);
y(:,1:n) = A;
for i = n+1:15000
    x = rand(d,1);
    y(:,i) = x./sum(x);
    [loss_fixed(i),gradient] = loss_fixed_action(y(:,i));
end
loss_f = min(loss_fixed);
[bestx,best_y] = find(loss_fixed == min(loss_fixed));
best = y(:,bestx);

% compute regret 
R = loss-loss_f;

%% plot of the strategy A and the coin data

% if you store the strategy in the matrix A (size d * n)
% this piece of code will visualize your strategy

figure
subplot(1,2,1);
plot(A')
legend(symbols_str)
title('rebalancing strategy OGD')
xlabel('date')
ylabel('investment action a_t')

subplot(1,2,2);
plot(s)
legend(symbols_str)
title('worth of coins')
xlabel('date')
ylabel('USD')
