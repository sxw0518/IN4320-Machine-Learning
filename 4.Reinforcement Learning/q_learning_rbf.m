function  theta_new = q_learning_rbf(theta,alpha,gamma,epsilon,rho,action)
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
for i = 1:size(theta,2)
    dev = rbf(s_new,rho);
    theta_new(action_num,i) = theta(action_num,i)+alpha*(r + gamma*Q_s_new_max - theta(action_num,:)*dev)*dev(i);
end
%theta_new(action_num,:)=theta(action_num,:)+alpha*(r + gamma*Q_s_new_max-theta(action_num,:)*rbf(s_new,rho)).*rbf(s_new,rho);
end