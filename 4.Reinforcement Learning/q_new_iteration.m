function q = q_new_iteration(q_previous,gamma,state,action,reward)
q = q_previous;
state_num = size(state,2);
action_num = size(action,2);
for num = 2:state_num-1
    for a = 1:action_num
        s = state(num);
        s_new = state(num)+action(a);
        q(a,s) = 0.7*(reward(s_new) + gamma*max(q(1,s_new),q(2,s_new)))+0.3*(reward(s)+gamma*max(q(1,s),q(2,s)));
    end
end
end