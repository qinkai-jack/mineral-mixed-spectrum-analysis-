function X = lrelu(P)
 X = max(P,0.01*P);
end
