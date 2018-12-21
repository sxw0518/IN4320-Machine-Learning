%%
% initialization
state = (1:6);
action = [-1,1];
reward = [1,0,0,0,0,5];
q = rand(2,6);
gamma = 0.5;
q_op = [0,1,0.5,0.625,1.25,0;0,0.625,1.25,2.5,5,0];
% rho = 0.1;
% theta = zeros(2,6);
% theta_new = theta;
%%
% Q-learning
alpha = (0:0.05:1);
epsilon = (0:0.05:1);
%%
iteration = 3000;
%%
q_final = zeros(size(q_op,1),size(q_op,2),size(alpha,2),size(epsilon,2),iteration);
error = zeros(size(alpha,2),size(epsilon,2),iteration);
%%
for i = 1:size(alpha,2)
    for j = 1:size(epsilon,2)
%         q = zeros(2,6);
        for p = 1:iteration
             s = 1.5 + 4*rand;
             rand_num = rand;
             if rand_num >= epsilon(j)
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
                 s_new = state(s,a(action_num));
             end 
             r = reward(s,s_new);
             Q_s_new_left = theta(1,:)*rbf(state(s_new,-1),rho);
             Q_s_new_right = theta(2,:)*rbf(state(s_new,1),rho);
             Q_s_new_max = max(Q_s_new_left,Q_s_new_right);
             for k = 1:size(theta,2)
                 dev = rbf(s_new,rho);
                 theta_new(action_num,k) = theta(action_num,k)+alpha(i)*(r + gamma*Q_s_new_max - theta(action_num,:)*dev)*dev(k);
             end 
             theta = theta_new;
             for m = 2:5
                 for n = 1:2
                     q_final(n,m,i,j,p) = theta(n,:)*rbf(m,rho);
                 end
             end
             if p>1
                 error(i,j,p-1) = norm(reshape(q_final(:,:,i,j,p),1,12)-reshape(q_final(:,:,i,j,p-1),1,12));
             end
%             q_previous = q;
%             q = Q_new_learning(q_previous,eta(j),alpha(i),gamma,s,a,r);
%             q_final(:,:,i,j,p) = q;
%             error(i,j,p) = norm(reshape(q,1,12)-reshape(q_op,1,12));

        end
    end
end
%%
figure;
subplot(2,2,1);
plot(reshape(error(3,2,:),1,iteration));
hold on;
plot(reshape(error(3,5,:),1,iteration));
plot(reshape(error(3,7,:),1,iteration));
plot(reshape(error(3,13,:),1,iteration));
plot(reshape(error(3,16,:),1,iteration));
plot(reshape(error(3,18,:),1,iteration));
xlabel('number of interactions');
ylabel('difference (2-norm)');
legend('\epsilon = 0.05','\epsilon = 0.2','\epsilon = 0.3','\epsilon = 0.6','\epsilon = 0.75','\epsilon = 0.85');
title('\alpha = 0.1')
subplot(2,2,2);
plot(reshape(error(6,2,:),1,iteration));
hold on;
plot(reshape(error(6,5,:),1,iteration));
plot(reshape(error(6,7,:),1,iteration));
plot(reshape(error(6,13,:),1,iteration));
plot(reshape(error(6,16,:),1,iteration));
plot(reshape(error(6,18,:),1,iteration));
xlabel('number of interactions');
ylabel('difference (2-norm)');
legend('\epsilon = 0.05','\epsilon = 0.2','\epsilon = 0.3','\epsilon = 0.6','\epsilon = 0.75','\epsilon = 0.85');
title('\alpha = 0.25')
subplot(2,2,3);
plot(reshape(error(11,2,:),1,iteration));
hold on;
plot(reshape(error(11,5,:),1,iteration));
plot(reshape(error(11,7,:),1,iteration));
plot(reshape(error(11,13,:),1,iteration));
plot(reshape(error(11,16,:),1,iteration));
plot(reshape(error(11,18,:),1,iteration));
xlabel('number of interactions');
ylabel('difference (2-norm)');
legend('\epsilon = 0.05','\epsilon = 0.2','\epsilon = 0.3','\epsilon = 0.6','\epsilon = 0.75','\epsilon = 0.85');
title('\alpha = 0.5')
subplot(2,2,4);
plot(reshape(error(18,2,:),1,iteration));
hold on;
plot(reshape(error(18,5,:),1,iteration));
plot(reshape(error(18,7,:),1,iteration));
plot(reshape(error(18,13,:),1,iteration));
plot(reshape(error(18,16,:),1,iteration));
plot(reshape(error(18,18,:),1,iteration));
xlabel('number of interactions');
ylabel('difference (2-norm)');
legend('\epsilon = 0.05','\epsilon = 0.2','\epsilon = 0.3','\epsilon = 0.6','\epsilon = 0.75','\epsilon = 0.85');
title('\alpha = 0.85')
%%
figure;
subplot(2,2,1);
plot(reshape(error(2,3,:),1,iteration));
hold on;
plot(reshape(error(5,3,:),1,iteration));
plot(reshape(error(7,3,:),1,iteration));
plot(reshape(error(13,3,:),1,iteration));
plot(reshape(error(16,3,:),1,iteration));
plot(reshape(error(18,3,:),1,iteration));
xlabel('number of interactions');
ylabel('difference (2-norm)');
legend('\alpha = 0.05','\alpha = 0.2','\alpha = 0.3','\alpha = 0.6','\alpha = 0.75','\alpha = 0.85');
title('\epsilon = 0.1')
subplot(2,2,2);
plot(reshape(error(2,5,:),1,iteration));
hold on;
plot(reshape(error(5,5,:),1,iteration));
plot(reshape(error(7,5,:),1,iteration));
plot(reshape(error(13,5,:),1,iteration));
plot(reshape(error(16,5,:),1,iteration));
plot(reshape(error(18,5,:),1,iteration));
xlabel('number of interactions');
ylabel('difference (2-norm)');
legend('\alpha = 0.05','\alpha = 0.2','\alpha = 0.3','\alpha = 0.6','\alpha = 0.75','\alpha = 0.85');
title('\epsilon = 0.25')
subplot(2,2,3);
plot(reshape(error(2,11,:),1,iteration));
hold on;
plot(reshape(error(5,11,:),1,iteration));
plot(reshape(error(7,11,:),1,iteration));
plot(reshape(error(13,11,:),1,iteration));
plot(reshape(error(16,11,:),1,iteration));
plot(reshape(error(18,11,:),1,iteration));
xlabel('number of interactions');
ylabel('difference (2-norm)');
legend('\alpha = 0.05','\alpha = 0.2','\alpha = 0.3','\alpha = 0.6','\alpha = 0.75','\alpha = 0.85');
title('\epsilon = 0.5')
subplot(2,2,4);
plot(reshape(error(2,18,:),1,iteration));
hold on;
plot(reshape(error(5,18,:),1,iteration));
plot(reshape(error(7,18,:),1,iteration));
plot(reshape(error(13,18,:),1,iteration));
plot(reshape(error(16,18,:),1,iteration));
plot(reshape(error(18,18,:),1,iteration));
xlabel('number of interactions');
ylabel('difference (2-norm)');
legend('\alpha = 0.05','\alpha = 0.2','\alpha = 0.3','\alpha = 0.6','\alpha = 0.75','\alpha = 0.85');
title('\epsilon = 0.85')
%%
time = [1:0.01:6];
y = 1;
for t = 1:0.01:6
    left(y) = theta(1,:)*rbf(t,rho);
    right(y) = theta(2,:)*rbf(t,rho);
    y = y+1;
end
figure;
plot(time,left);
hold on;
plot(time,right);
%%
plot(reshape(error(9,1,:),1,iteration-1));
legend('Convergence');
%% exercise 7
for i = 2:size(alpha,2)
    for j = 1:size(epsilon,2)
        q = rand(2,6);
        for p = 1:iteration
%              q = sarsa(q,epsilon(j),alpha(i),gamma,a,r);
s = randi([2 5],1,1);
random_num = rand;
if random_num >= epsilon(j)
    if q(1,s)>q(2,s)
        a = 1;
        s_new = s + action(a);
    else
        a = 2;
        s_new = s + action(a);
    end
else
    a = 1 + randi([0 1],1,1);
    s_new =s + action(a);
end
while s_new<6 && s_new>1
    r = reward(s_new);
    rand_num = rand;
    if rand_num >= epsilon(j)
        if q(1,s_new)>q(2,s_new)
            a_new = 1;
        else
            a_new = 2;
        end  
    else
        a_new = 1 + randi([0 1],1,1);
    end
    q(a,s) = q(a,s)+alpha(i)*(r+gamma*q(a_new,s_new)-q(a,s));
    s_new = s_new + action(a_new);
    a = a_new;
end
             q_final(:,:,i,j,p) = q;
             error(i,j,p) = norm(reshape(q,1,12)-reshape(q_op,1,12));
        end
    end
end
%%

