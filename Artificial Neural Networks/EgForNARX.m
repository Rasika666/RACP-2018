% This program demonstrates the use of the series-parallel architecture for 
%training an NARX network to model a dynamic system. 
% 
% The goal is to develop an NARX model for "magnetic levitation system". 
% First load the training data. 
% Use tapped delay lines with two delays for both the input and the output, 
%so training begins with the third data point. 
% There are two inputs to the series-parallel network, the u(t) sequence 
%and the y(t) sequence, so p is a cell array with two rows. 

load magdata
[u1,us] = mapminmax(u);
[y1,ys] = mapminmax(y); 
y = con2seq(y1); u = con2seq(u1);
p = [u(3:end);y(3:end)]; t = y(3:end);

% Then create the series-parallel NARX network using the function newnarxsp. 
% Use 10 neurons in the hidden layer and use trainbr for the training function. 

d1 = [1:2];
d2 = [1:2];
narx_net = newnarxsp({[-1 1],[-1 1]},d1,d2,[10 1],{'tansig','purelin'});
narx_net.trainFcn = 'trainbr';
narx_net.trainParam.show = 10;
narx_net.trainParam.epochs = 600;

% Now you are ready to train the network. 
% First you need to load the tapped delay lines with the initial inputs and 
%outputs. 
% The following commands illustrate these steps. 

for k=1:2,
  Pi{1,k}=u{k};
end
for k=1:2,
  Pi{2,k}=y{k};
end
narx_net = train(narx_net,p,t,Pi);

% You can now simulate the network and plot the resulting errors for the 
%series-parallel implementation. 

yp = sim(narx_net,p,Pi);
e = cell2mat(yp)-cell2mat(t);
plot(e)


% The result is displayed in the resultant plot. 
% You can see that the errors are very small. 
% However, because of the series-parallel configuration, these are errors 
%for only a one-step-ahead prediction. 
% A more stringent test would be to rearrange the network into the original 
%parallel form 
% and then to perform an iterated prediction over many time steps. 
% Now the parallel operation is demonstrated.

% There is a toolbox function (sp2narx) for converting NARX networks from 
%the series-parallel configuration, 
% which is useful for training, to the parallel configuration. 
% The following commands illustrate how to convert the network just trained 
%to parallel form 
% and then use that parallel configuration to perform an iterated prediction 
%of 900 time steps. 
% In this network you need to load the two initial inputs and the two 
%initial outputs as initial conditions. 

narx_net2 = sp2narx(narx_net);
y1=y(1700:2600); u1=u(1700:2600);
p1 = u1(3:end); t1 = y1(3:end);
for k=1:2,
  Ai1{1,k}=zeros(10,1);
  Ai1{2,k}=y1{k};
end
for k=1:2,
  Pi1{1,k}=u1{k};
end
yp1 = sim(narx_net2,p1,Pi1,Ai1);
plot([cell2mat(yp1)' cell2mat(t1)'])


% The resultant figure illustrates the iterated prediction. 
% The solid line is the actual position of the magnet, and the dashed line 
%is the position predicted by the NARX neural network. 
% Although the network prediction is noticeably different from the actual 
%response after 50 time steps, 
% the general behavior of the model is very similar to the behavior of the 
%actual system. 
% By collecting more data and training the network further, you can produce 
%an even more accurate result.

% In order for the parallel response to be accurate, it is important that 
%the network be trained 
% so that the errors in the series-parallel configuration are very small. 
% 
% You can also create a parallel NARX network, using the newnarx command, 
%and train that network directly. 
% Generally, the training takes longer, and the resulting performance is 
%not as good as that obtained with series-parallel training.