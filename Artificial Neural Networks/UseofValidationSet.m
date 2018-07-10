% The following sequence of commands demonstrates how to use the early stopping function. 
% 
% Create a simple test problem. For the training set, generate a noisy sine wave with input points ranging from -1 to 1 at steps of 0.05. 
p = [-1:0.05:1];
t = sin(2*pi*p)+0.1*randn(size(p));

% Generate the validation set. The inputs range from -1 to 1, as in the test set, but offset slightly. 
% To make the problem more realistic, also add a different noise sequence to the underlying sine wave. 
% Notice that the validation set is contained in a structure that contains both the inputs and the targets. 
val.P = [-0.975:.05:0.975];
val.T = sin(2*pi*val.P)+0.1*randn(size(val.P));

% Now create a 1-20-1 network, as in the previous example with regularization, and train it. 
% (Notice that the validation structure is passed to train after the initial input and layer conditions, which are null vectors in this case 
% because the network contains no delays. This example does not use a test set. (The test set structure would be the next argument in the call to train.) This example uses the training function traingdx, although early stopping can be used with any of the other training functions discussed in this chapter. 
net=newff([-1 1],[20,1],{'tansig','purelin'},'traingdx');
net.trainParam.show = 25;
net.trainParam.epochs = 300;
net.trainParam.mc=0.01;
net.trainParam.lr=0.003;
net = init(net);
[net,tr]=train(net,p,t,[],[],val);
