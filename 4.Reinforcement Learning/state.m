function s_new = state(s,a)
if s<1.5||s>=5.5
    s_new = s;
else
    s_new = s + a + random('Normal',0,0.1);
end
end