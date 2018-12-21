function q = q_iteration(q_previous,gamma,state,action,reward)
q = q_previous;
state_num = size(state,2);
action_num = size(action,2);
for num = 2:state_num-1
    for a = 1:action_num
        s = state(num);
        s_new = state(num)+action(a);
        q(a,s) = reward(s_new) + gamma*max(q(1,s_new),q(2,s_new));
    end
end

end