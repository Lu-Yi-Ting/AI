function R = RLSE(input,output,theta)
len = size(input,1); 
A = input;
y = output;
a = 10^9;
I = eye(size(theta,1));
P = a.*I;
for i=1:len
    P = P - P*transpose(A(i,:))*A(i,:)*P / (1+A(i,:)*P*transpose(A(i,:)));
    theta = theta + P*transpose(A(i,:))*(y(i)-A(i,:)*theta);
end
R = theta;