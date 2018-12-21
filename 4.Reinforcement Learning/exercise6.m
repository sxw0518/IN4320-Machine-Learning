gamma = 0.5;
alpha = 0.4;
epsilon = 0.7;
theta = zeros(2,6);
q = zeros(2,6);
rho = 0.1;
action = [-1,1];
theta_new = theta;
Q = zeros(2,6,10000)
%%
for i = 1:10000
    s = 1.5 + 4*rand;
    rand_num = rand;
    if rand_num >= epsilon
        s_left = state(s,-1);
        s_right = state(s,1);
        Q_s_left = theta(1,:)*rbf(s_left,rho);
        Q_s_right = theta(2,:)*rbf(s_right,rho);
        if Q_s_left > Q_s_right
            s_new = s_left;
            action_num = 1;
        else
            s_new = s_right;
            action_num = 2;
        end
    else
        action_num = randi([0 1],1,1)+1;
        s_new = state(s,action(action_num));
    end
    r = reward(s,s_new);
    Q_s_new_left = theta(1,:)*rbf(state(s_new,-1),rho);
    Q_s_new_right = theta(2,:)*rbf(state(s_new,1),rho);
    Q_s_new_max = max(Q_s_new_left,Q_s_new_right);
    for j = 1:size(theta,2)
        dev = rbf(s_new,rho);
        theta_new(action_num,j) = theta(action_num,j)+alpha*(r + gamma*Q_s_new_max - theta(action_num,:)*dev)*dev(j);
    end
    theta = theta_new;
    for p = 2:5
        for k = 1:2
            Q(k,p,i) = theta(k,:)*rbf(p,rho);
        end
    end
end
%%
