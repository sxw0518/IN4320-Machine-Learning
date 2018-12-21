function q = Q_new_learning(q_previous,eta,alpha,gamma,state,action,reward)
q = q_previous;
state_num = size(state,2);
for num = 2:state_num-1
    s = state(num);
    random_num = rand;
    if random_num >= eta
        if q(1,s)>q(2,s)
            a = 1;
        else
            a = 2;
        end
    else
        a = 1 + randi([0 1], 1, 1);
    end
    random_num = rand;
    if random_num >= 0.3
        s_new = state(num)+action(a);
    else
        s_new = state(num);
    end
    q(a,s) = q(a,s)+alpha*(reward(s_new) + gamma*max(q(1,s_new),q(2,s_new))-q(a,s));
end
end