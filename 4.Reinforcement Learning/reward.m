function r = reward(s,s_new)
if s_new >= 5.5 && s < 5.5 && s >= 1.5
    r = 5;
elseif s_new<1.5 && s >= 1.5 && s < 5.5
    r = 1;
else
    r = 0;
end
end