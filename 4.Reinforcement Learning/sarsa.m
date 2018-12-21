function q = sarsa(q_previous,eta,alpha,gamma,action,reward)
q = q_previous;
s = randi([2 5],1,1);
random_num = rand;
if random_num >= eta
    if q(1,s)>q(2,s)
        a = 1;
        s_new = s+ action(a);
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
    if rand_num >= eta
        if q(1,s_new)>q(2,s_new)
            a_new = 1;
        else
            a_new = 2;
        end  
    else
        a_new = 1 + randi([0 1],1,1);
    end
    q(a,s) = q(a,s)+alpha*(r+gamma*q(a_new,s_new)-q(a,s));
    s_new = s_new + action(a_new);
    a = a_new;
end
end